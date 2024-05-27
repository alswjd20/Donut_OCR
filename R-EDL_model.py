"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import os
import re
from typing import Any, List, Optional, Union

import numpy as np
import PIL
import timm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers import MBartConfig, MBartForCausalLM, XLMRobertaTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import wandb

# R_EDL, relaxed edl 논문 적용한 버전 
# r-edl



class SwinEncoder(nn.Module):
    r"""
    Donut encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations as a Donut Encoder

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None

        # weight init with swin
        if not name_or_path:
            swin_state_dict = timm.create_model("swin_base_patch4_window12_384", pretrained=True).state_dict()
            new_swin_state_dict = self.model.state_dict()
            for x in new_swin_state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                    pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                    new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
                else:
                    new_swin_state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(new_swin_state_dict)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("SwinEncoder 클래스 forward 함수 호출됨")
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        return x

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))


class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
        self, decoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            "hyunwoongko/asian-bart-ecjk" if not name_or_path else name_or_path
        )
        # print("tokenizer 확인 : ", self.tokenizer) # tokenizer 확인 :  PreTrainedTokenizer(name_or_path='naver-clova-ix/donut-base-finetuned-cord-v2', vocab_size=57522, model_max_len=1000000000000000019884624838656, is_fast=False, padding_side='right', ...})

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
            )
        )
        self.model.forward = self.forward  #  to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.add_special_tokens(["<sep/>"])  # <sep/> is used for representing a list in a JSON
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id # 1
        # print("pad_token_id 확인 : ", self.tokenizer.pad_token_id) # 그냥 1이라 나옴 

        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.softplus = nn.Softplus()
        self.target_con = 1.0
        self.kl_c = -1.0
        self.fisher_c = 1.0
        self.lamb1 = 1.0
        self.lamb2 = 1.0

        self.loss_mse = torch.tensor(0.0)
        self.loss_var = torch.tensor(0.0)
        self.loss_kl = torch.tensor(0.0)
        self.loss_fisher = torch.tensor(0.0)

        # weight init with asian-bart
        if not name_or_path:
            bart_state_dict = MBartForCausalLM.from_pretrained("hyunwoongko/asian-bart-ecjk").state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if x.endswith("embed_positions.weight") and self.max_position_embeddings != 1024:
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    new_bart_state_dict[x] = bart_state_dict[x][: len(self.tokenizer), :]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)
        



    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, past_key_values=None, past=None, use_cache: bool = None, attention_mask: torch.Tensor = None):

        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        # for compatibility with transformers==4.11.x
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }

        return output
    
    
    
    def one_hot_embedding(self, labels, num_classes):
        y = torch.eye(num_classes, device = 'cpu')
        labels_cpu = labels.cpu()
        # print(labels_cpu.device)

        # -100인 것들은 0으로.. 
        labels_cpu = torch.where(labels_cpu == -100, torch.zeros_like(labels_cpu), labels_cpu)
        # breakpoint()

        # y[labels_cpu]은 그니까, -100이었던 것들은, 0번째 인덱스가 1로 된 거지. 
        # 뒷 부분은 다 0번째 인덱스가 1로 된 원 핫이라 보면 된다 
        return y[labels_cpu] 
    
    
    
    def compute_mse(self, labels_1hot, evidence):
        num_classes = evidence.shape[-1]

        gap = labels_1hot - (evidence + self.lamb2) / \
              (evidence + self.lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + self.lamb2 * num_classes)

        loss_mse = gap.pow(2).sum(-1)

        return loss_mse.mean()

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        target_alphas = torch.ones_like(alphas) * target_concentration

        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss
    

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
        epoch : int = None,
    ):
        # breakpoint()
        # print("BARTDecoder 클래스 forward 함수 호출됨")
        # print("호출됨") # 이미지에 따른 각 prob_list의 개수만큼 -> forward 함수가 호출되는 것 확인  
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
        # print("BARTDecoder 클래스의 forward 함수안으로 들어왔음.") # test 할 때도 실행됨 <- 여러 번 연속으로 출력되는 걸 보니, 연속으로 여러 번 호출되는 듯 
        # 여러 번 출력되는게 지저분해보여서 그냥 주석처리 했음 

        """lm_head를 이용해서 계산됨 -> 이 logit이 inference 할땐 어찌 저찌 계산되어 decoder_output.scores로 나오는 걸 확인했음."""
        logits = self.model.lm_head(outputs[0]) # torch.Size([1, 767, 57580]) # grad_fn=<UnsafeViewBackward0> 
        """내가 추가해준 edl_layer을 이용해서 계산되었음 -> 이 alphas가 inference할 때 edl_scores로 나오도록 코드 작성해줬음."""
        edl_logits = self.model.edl_layer(outputs[0]) # 4번을 위해 추가 # torch.Size([1, 767, 57580]) # grad_fn=<UnsafeViewBackward0>
        # alphas = torch.nn.functional.relu(edl_logits[0]) + 1.0 # 결국, 여기서 alphas는 edl_layer을 통해 나오는 값을 갖고 계산된게 맞네. 


        # clf_type = 'softplus' # R-EDL은 디폴트가 softplus더라 
        evi_alpha = self.softplus(edl_logits) + 1.0

        evidence = self.softplus(edl_logits[0])
        evi_alp_ = evidence + self.lamb2
        

        loss = None
        if labels is not None: # train할때만 실행됨 
            
            print("이게 evi_alp_ - 학습이 진행됨에 따라 어떻게 변하고 있어? : ", evi_alp_)

            # 기존 donutmodel에서 학습시키던 loss function : cross_entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # igenore_id값이 -100으로 들어가는구나
            ce_loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1)) # vocab_size는 57580, self.model.config.vocab_size) : 57580
            # labels.shape : torch.Size([1, 767]) 2중 리스트 형태 -> labels.view(-1).shape : torch.Size([767]) 리스트 하나 형태 


            # ignore_id 하는 부분 
            labels_ignored = labels[labels != -100] # labels_ignored.shape : torch.Size([33])
            y = self.one_hot_embedding(labels_ignored.view(-1), self.model.config.vocab_size) # vocab_size # 57580 -> y.shape : torch.Size([33, 57580])
            evi_ignored = evidence[labels[0] != -100] # evi_alp_ignored.shape : torch.Size([33, 57580])

            y = y.to('cuda')

            self.loss_mse = self.compute_mse(y.float(), evi_ignored)

            kl_alpha = evi_ignored * (1 - y.float()) + self.lamb2
            self.loss_kl = self.compute_kl_loss(kl_alpha, self.lamb2)


            if self.kl_c == -1:
                regr = np.minimum(1.0, epoch / 10.)
                edl_loss = self.loss_mse + regr * self.loss_kl
            else:
                edl_loss = self.loss_mse + self.kl_c * self.loss_kl
            

            loss = ce_loss + edl_loss
            print("ce loss : ", ce_loss)
            print("edl_loss : ", edl_loss)
            
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
             
             
        # logit은 logit대로 내게 하고, alpha를 추가해볼까? 싶어서  
        return ModelOutput(
            loss=loss,
            logits=logits,
            edl_logits=edl_logits,
            # alphas=alphas,
            alphas = evi_alpha,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight
    

class DonutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "donut"

    def __init__(
        self,
        input_size: List[int] = [2560, 1920],
        align_long_axis: bool = False,
        window_size: int = 10,
        encoder_layer: List[int] = [2, 2, 14, 2],
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 1536,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer

        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.name_or_path = name_or_path


class DonutModel(PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = DonutConfig
    base_model_prefix = "donut"

    def __init__(self, config: DonutConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
        )
        self.decoder = BARTDecoder(
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            name_or_path=self.config.name_or_path,
        )

        # print("여기서도 값이 동일하나? 뭐가 먼저 실행되는거야? ", self.decoder.model.lm_head) # Linear(in_features=1024, out_features=57525, bias=False) <- 이게 먼저 실행됨 


        self.current_img_idx = 0
        self.current_epoch = 0

        
        self.edl_train_loss = dict()
        self.train_loss = []


    def forward(self, image_tensors: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor):
        # print("DonutModel 클래스 forward 함수 호출됨") # 이 부분은 test 할 땐 실행 안됨!!! -> inference가 실행됨 

        print("현재 처리중인 이미지 인덱스 : ", self.current_img_idx)
        print("현재 epoch : ", self.current_epoch)


        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """

        encoder_outputs = self.encoder(image_tensors) # image tensor이 encoder(SwinEncoder)에 들어가네?
        # 그래서 encoder outputs으로 나오고 
        decoder_outputs = self.decoder( # decoder(BARTDecoder)에 input으로 들어가는 것들
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs, # encoder outputs이 decoder에 input으로 들어감 
            labels=decoder_labels,
            epoch = self.current_epoch,
        )

        # print("@@@@@@@@@@@@@@@ 이게 출력되긴 해? : ", decoder_outputs.alphas) # 잘 출력되는데? 
        # breakpoint()

        
        """5번, 6번, freeze 시키는 경우 """
        print("####### loss값 계산한 직후 - freeze 되었는지 확인") # 진짜 한 번 확인해볼까? 잘 freeze 되었는지  -> 잘 되긴 했는데, 혹시 모르니 
        for param in self.parameters():
            param.requires_grad = False
        for last_param in self.decoder.model.lm_head.parameters():
            last_param.requires_grad = True
        for last_param in self.decoder.model.edl_layer.parameters():
            last_param.requires_grad = True  # 마지막 layer만 freeze 시키지 않음..
        for name, all_param in self.named_parameters():
            if all_param.requires_grad == True : 
                print("######## loss값 계산한 직후 - freeze 안된 layer : ", name)
                # print("######## loss값 계산한 직후 - freeze 안된 layer : ", all_param.shape) # torch.Size([57580, 1024])
                # print("######### loss값 계산한 직후 - freeze 안된 layer : ", all_param.requires_grad) 
        
        
        with torch.no_grad():
            loss = decoder_outputs.loss
            print("이게 이번 loss : ", loss)
            loss_cpu = loss.cpu()
            self.train_loss.append(loss_cpu.item())
            print("이게 loss 전체 리스트임 : ", self.train_loss)
        

        if self.current_img_idx == 799:
            # print("리스트의 value 개수, length : ", len(self.train_loss))
            self.edl_train_loss[self.current_epoch] = sum(self.train_loss) / len(self.train_loss)

            with open('train_redl_loss_7_0527_redl_nfreeze.json','w') as f:
                json.dump(self.edl_train_loss, f)
                print("json 파일 생성")
            
            self.current_epoch += 1
            self.current_img_idx = 0
            self.train_loss = []
            torch.cuda.empty_cache() 
        
        else:
            self.current_img_idx += 1


        return decoder_outputs

    
    def inference(
        self,
        image: PIL.Image = None,
        prompt: str = None,
        image_tensors: Optional[torch.Tensor] = None,
        prompt_tensors: Optional[torch.Tensor] = None,
        return_json: bool = True,
        return_attentions: bool = False,
    ):
        print("donutmodel 클래스 inference 함수 호출됨!!!! 0504 수정.. ") # 이 부분은 test 할 때 실행됨 
        # test시, 이 부분은 이미지당 한 번만 호출되고, BARTDecoder 클래스의 forward 함수는 여러 번 연속으로 호출됨 
        # test시, DonutModel 클래스의 forward 함수는 호출되지 않고, 대신 이 inference 함수가 호출됨 

        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width) # Batch size가 디폴트로 1로 저장되어있는것, (이미지 한 장씩)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)


        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
            output_scores=True,
            output_edl_scores=True # 4번을 위해 추가 
        )

        print("출력된 decoder_output.sequences : ", decoder_output.sequences)

        edl_posterior_scores = []
        for score_tensor in decoder_output.edl_scores: # edl_layer에서 나온 output들(edl_scores)
            # print(score_tensor.shape) # torch.Size([1, 57580])
            
            """확률을 계산하는 두 번째 방법"""
            alpha_tensor = score_tensor.to(torch.float64) 
            edl_posterior_scores.append(torch.nn.functional.softmax(alpha_tensor[0], dim=0)) 


            """확률을 계산하는 첫 번째 방법을 구하기 위해 잠시 주석처리...
            alpha_tensor = torch.nn.functional.relu(score_tensor) + 1 # torch.Size([1, 57580]) 이건 해줄 필요 없어. 이미 alpha tensor야. 
            alpha_zero = torch.sum(alpha_tensor, dim=1, keepdim=True) # dtype=torch.float16, torch.Size([1, 1])
            prob_list = (alpha_tensor / float(alpha_zero))[0] # alpha값이 같은 1.000이라도, alpha_zero 값이 다르기 때문에 probability가 달라짐. 
            edl_posterior_scores.append(prob_list) """
            


        """ 아래는 그냥 softmax로 probability를 계산한 일반 버전 
        이건 맞음. decoder_output.scores에 softmax 취하면 됨. utils.py 확인해보면, scores는 outputs.logits에서 나온게 맞고, 
        logits는 lm_head에서 나온게 맞음. """
        softmax_scores = []
        logit_scores = []
        for score_tensor in decoder_output.scores: # 이건 lm_head에서 나온 scores인거지 
            # print("소프트 맥스 취한 값 확인 : ", torch.nn.functional.softmax(score_tensor[0], dim=0))
            # print("합이 1인지 확인 : ",torch.nn.functional.softmax(score_tensor[0], dim=0).sum()) # -> 맞음. 
            score_tensor = score_tensor.to(torch.float64)
            logit_scores.append(score_tensor[0])
            softmax_scores.append(torch.nn.functional.softmax(score_tensor[0], dim=0))
            # score_list = torch.nn.functional.softmax(score_tensor[0], dim=0).tolist() # tensor -> list로 
            
            # breakpoint()

        # breakpoint() # decoder_output.sequences의 length랑 decoder_output.scores의 length랑 비교
        # len(edl_posterior_scores) = 55 
        # len(decoder_output.sequences[0]) = 56 

        # => 길이가 달라.. len(decoder_output.sequences[0]) = 59, len(decoder_output.scores) = 58


        output = {"predictions": list()}
        model_pred_ids = decoder_output.sequences

        # 여기 batch_decode에서 id -> token
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):

            print("seq : ", seq)

            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                # print("token2json에서 반환되어 나온 결과 : ", self.token2json(seq))
                output["predictions"].append(self.token2json(seq)) # seq 출력해봤으면 알겠지만 토큰인데 -> 이를 json으로 바꿔주는 듯
            else:
                output["predictions"].append(seq)


        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }


        output["logits"] = decoder_output.scores
        
        # edl은 alpha를 갖고 확률을 계산하더라. 
        output["prob_alpha"] = edl_posterior_scores
        output["prob_softmax"] = softmax_scores # 이건 기존 

        output["model_pred_id"] = model_pred_ids
        # print("output 확인 : ", output) # 확인해보면, 'prob' : [tensor([]), tensor([]), tensor([]), ...] 
        # 이렇게 리스트 안에 여러 tensor([])이 들어가있는 형태 

        return output

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value

                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in self.decoder.tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                            # breakpoint()
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(pretrained_model_name_or_path, revision="official", *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model

    # 아래는 내가 추가한 부분 (# 4번을 위해 추가 )
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            print("~~~~~~~~~~~~~~~~~~~0516 아니 이 부분이 실행되긴 하는거야?~~~~~~~~~~~~~~~~~~~~~`")
            print("~~~~~~~~~~ weight initialize 시키는 부분 실행됩니다. 이 부분은 train할 때만 실행됨 ~~~~~~~~~~~") # 혹시 이 부분이 test 때도 실행되나? 그게 문제인건가? 
            # breakpoint()
            # 이 부분은 train 할 때만 실행되고, test때는 실행 안됨 
            module.weight.data.normal_(mean=0.0, std=0.05)  # 예시로 표준 편차를 0.02로 설정
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.05)  # 예시로 표준 편차를 0.02로 설정
            if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    
