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
import tensorflow as tf

# uncertainty quantification for classification 논문 적용
# 4_edl 마지막으로 수정.. 0529... => 9_edl


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
    

    def loglikelihood_loss(self, y, alpha): # y : torch.Size([33, 57580]), alpha : torch.Size([33, 57580])

        # breakpoint()

        S = torch.sum(alpha, dim=1, keepdim=True) # torch.Size([767, 1]) -> [33, 1]

        # breakpoint()
        
        squared_error = (torch.tensor(y).to('cuda') - (alpha / S)) ** 2 
        variance = alpha * (S - alpha.float()) / (S * S * (S + 1))

        # breakpoint() # squared_error랑, variance에 requires_grad = True 붙어있어? 
        # squared_error : grad_fn=<PowBackward0>
        # variance : grad_fn=<DivBackward0>

        loglikelihood_err = torch.sum(squared_error, dim=1, keepdim=True) # grad_fn=<SumBackward1>
        loglikelihood_var = torch.sum(variance, dim=1, keepdim=True) # grad_fn=<SumBackward1>
        
        """기존 코드
        # loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        # loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True) """

        loglikelihood = loglikelihood_err + loglikelihood_var
        # print("뒤에 인덱스인 애들은 다 0으로 뜸 : ", loglikelihood) grad_fn=<AddBackward0>
        
        return loglikelihood
    
    def relu_evidence(self, output):
        return torch.nn.functional.relu(output)

    def softplus_evidence(self, output):
        # breakpoint()
        return self.softplus(output)
    
    def exp_evidence(self, output): 
        clipped_output = torch.clamp(output, min=-10, max=10)
        return torch.exp(clipped_output)
        
    
    def one_hot_embedding(self, labels, num_classes, ignore_index): # len(labels) : 767
        y = torch.eye(num_classes, device = 'cpu') # torch.Size([57580, 57580])
        labels_cpu = labels.cpu()
        # print(labels_cpu.device)

        # breakpoint() # ignore_id도 여기서 같이 해버리자. 
        labels_cpu = labels_cpu[labels_cpu != ignore_index]

        # breakpoint()
        
        return y[labels_cpu] # torch.Size([33, 57580])
    
    
    def kl_divergence(self, alpha, num_classes):

        ones = torch.ones([1, num_classes], dtype=alpha.dtype, device = alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl
    
    def edl_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step):
        y = y.to('cuda')
        alpha = alpha.to('cuda')

        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        # 마찬가지로, kl term은 우선 무시
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)

        return A + kl_div
    
    def edl_loss_ignore_kl(self, func, y, alpha, epoch_num, num_classes, annealing_step):
        y = y.to('cuda')
        alpha = alpha.to('cuda')

        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        # kl term은 우선 무시
        # annealing_coef = torch.min(
        #     torch.tensor(1.0, dtype=torch.float32),
        #     torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        # )
        # kl_alpha = (alpha - 1) * (1 - y) + 1
        # kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)

        return A # + kl_div
    

    def mse_loss(self, y, alpha, epoch_num, num_classes, annealing_step):

        y = y.to('cuda')
        alpha = alpha.to('cuda')

        loglikelihood = self.loglikelihood_loss(y, alpha)

        """ 
        일단 아래가 원래 코드.. 
        kl_alpha = (alpha - 1) * (1 - torch.tensor(y)).to('cuda') + 1 # torch.Size([767, 57580])
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes) # torch.Size([767, 1])
        """
        # 마찬가지로, kl term은 우선 무시
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float16),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float16),
        )

        kl_alpha = (alpha - 1) * (1 - torch.tensor(y)).to('cuda') + 1 # torch.Size([33, 57580])
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes) # torch.Size([33, 1])

        return loglikelihood + kl_div
    
    def mse_loss_ignore_kl(self, y, alpha, epoch_num, num_classes, annealing_step):

        y = y.to('cuda')
        alpha = alpha.to('cuda')

        loglikelihood = self.loglikelihood_loss(y, alpha)

        # 마찬가지로, kl term은 우선 무시
        # annealing_coef = torch.min(
        #     torch.tensor(1.0, dtype=torch.float16),
        #     torch.tensor(epoch_num / annealing_step, dtype=torch.float16),
        # )

        # kl_alpha = (alpha - 1) * (1 - torch.tensor(y)).to('cuda') + 1 # torch.Size([33, 57580])
        # kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes) # torch.Size([33, 1])

        return loglikelihood # + kl_div
    
    def edl_log_loss(self, output, target, epoch_num, num_classes, annealing_step, real_num_id, ignore_kl, evidence_type):

        output_cuda = output.cuda()

        if evidence_type == 'relu':
            evidence = self.relu_evidence(output_cuda)
            alpha = evidence + 1.0

        elif evidence_type == 'softplus':
            evidence = self.softplus_evidence(output_cuda)
            alpha = evidence + 1.0

        elif evidence_type == 'exp':
            evidence = self.exp_evidence(output_cuda)
            alpha = evidence + 1.0
        
        else : 
            print("evidence type : relu, softplus, exp 중 하나")

        if ignore_kl == False:
            loss = torch.sum(
                self.edl_loss(
                    torch.log, target, alpha, epoch_num, num_classes, annealing_step)
            )/ float(real_num_id)

        elif ignore_kl == True:
            loss = torch.sum(
                self.edl_loss_ignore_kl(
                    torch.log, target, alpha, epoch_num, num_classes, annealing_step)
            )/ float(real_num_id)
        
        else : 
            print("그래서 kl ignore 할거야 말거야")

        # 이 아래가 원래 loss...
        # loss = torch.mean(
        #     self.edl_loss(
        #         torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        #     )
        # )
        return loss
    
    def edl_digamma_loss(self, output, target, epoch_num, num_classes, annealing_step, real_num_id, ignore_kl, evidence_type):

        output_cuda = output.cuda()

        if evidence_type == 'relu':
            evidence = self.relu_evidence(output_cuda)
            alpha = evidence + 1.0

        elif evidence_type == 'softplus':
            evidence = self.softplus_evidence(output_cuda)
            alpha = evidence + 1.0

        elif evidence_type == 'exp':
            evidence = self.exp_evidence(output_cuda)
            alpha = evidence + 1.0
        
        else : 
            print("evidence type : relu, softplus, exp 중 하나")

        if ignore_kl == False:
            loss = torch.sum(
                self.edl_loss(
                    torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
            ) / float(real_num_id)

        elif ignore_kl == True: 
            loss = torch.sum(
                self.edl_loss_ignore_kl(
                    torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
            ) / float(real_num_id)
        
        else : 
            print("그래서 ignore kl 할거야 말거야")

        # 이 아래가 원래 loss
        # loss = torch.mean(
        # edl_loss(
        #     torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        # )
        # )

        return loss

    def edl_mse_loss(self, output, target, epoch_num, num_classes, annealing_step, real_num_id, ignore_kl, evidence_type):

        output_cuda = output.cuda()

        if evidence_type == 'relu':
            evidence = self.relu_evidence(output_cuda)
            alpha = evidence + 1.0

        elif evidence_type == 'softplus':
            evidence = self.softplus_evidence(output_cuda)
            # breakpoint()
            alpha = evidence + 1.0

        elif evidence_type == 'exp':
            evidence = self.exp_evidence(output_cuda)
            alpha = evidence + 1.0
        
        else : 
            print("evidence type : relu, softplus, exp 중 하나")


        if ignore_kl == False:
            loss = torch.sum(
                self.mse_loss(target, alpha, epoch_num, num_classes, annealing_step)
            ) / float(real_num_id)

        elif ignore_kl == True:
            loss = torch.sum(
                self.mse_loss_ignore_kl(target, alpha, epoch_num, num_classes, annealing_step)
            ) / float(real_num_id)

        else : 
            print("ignore_kl 할 건지 말건지 선택")
        
        # 이 아래가 원래 loss...
        # loss = torch.mean(
        #     self.mse_loss(target, alpha, epoch_num, num_classes, annealing_step, ignore_index, labels)
        # )

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
        alphas = torch.nn.functional.relu(logits) + 1 # grad_fn=<AddBackward0>   
      

        loss = None
        if labels is not None: # train할때만 실행됨  # labels.shape : torch.Size([1, 767])
            print("이게 alphas - 학습이 진행됨에 따라 어떻게 변하고 있어? : ", alphas)

            # 기존 donutmodel에서 학습시키던 loss function : cross_entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # igenore_id값이 -100으로 들어가는구나
            ce_loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1)) # vocab_size는 57580
            

            # edl_loss로 학습시키려면
            real_num_id = len(labels[0][labels[0] != -100].tolist())

            # 이 함수 안에서 ignore index 됨 
            y = self.one_hot_embedding(labels.view(-1), self.model.config.vocab_size, ignore_index = -100) # vocab_size # 57580
            # y.shape : torch.Size([33, 57580]) # 이미 ignore_index 된 상태 

            logits_ = logits.view(-1, self.model.config.vocab_size) # logits_.shape : torch.Size([767, 57580])
            logits_ignored_id = logits_[labels[0] != -100] # ignore_index 되었음 - shape : torch.Size([33, 57580])
            # breakpoint()

            # edl_mse_loss, edl_log_loss, edl_digamma_loss 3 중 하나 선택, 
            # ignore_kl = True로 하면, kl term이 무시됨
            # evidence_type = 'relu' or 'softplus' or 'exp' 중 하나 선택
            edl_loss = self.edl_log_loss( # self.model.config.vocab_size = 57580
                logits_ignored_id, y.float(), epoch, self.model.config.vocab_size, annealing_step = 10, real_num_id = real_num_id, ignore_kl = True, evidence_type = 'exp',
            )
            
            loss = ce_loss + edl_loss
            # loss = edl_loss

            print("ce loss : ", ce_loss)
            print("edl_loss : ", edl_loss)
            
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
             
             
        return ModelOutput(
            loss=loss,
            logits=logits,
            alphas=alphas,
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


        with torch.no_grad():
            loss = decoder_outputs.loss
            print("이게 이번 loss : ", loss)
            loss_cpu = loss.cpu()
            self.train_loss.append(loss_cpu.item())
            print("이게 loss 전체 리스트임 : ", self.train_loss)
        

        if self.current_img_idx == 799:
            # print("리스트의 value 개수, length : ", len(self.train_loss))
            self.edl_train_loss[self.current_epoch] = sum(self.train_loss) / len(self.train_loss)

            with open('train_loss_edl_loss_9_0611_edl_log_exp_nk_full_tr.json','w') as f:
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
            output_scores=True
        )

        print("decoder_output 나왔다...")

        print("출력된 decoder_output.sequences : ", decoder_output.sequences)
        # sequences = tensor([[57579, 57526, 57528, 20220, 38946,  4107, <- 이런 식으로 모델이 예측한 id값들이 나옴
            

        softmax_scores = []
        logit_scores = []
        for score_tensor in decoder_output.scores: # 이건 lm_head에서 나온 scores
            # breakpoint() # score_tensor.shape : torch.Size([1, 57580])
            
            score_tensor = score_tensor.to(torch.float64)
            logit_scores.append(score_tensor[0])
            softmax_scores.append(torch.nn.functional.softmax(score_tensor[0], dim=0)) 


        output = {"predictions": list()}
        model_pred_ids = decoder_output.sequences

        logit_stack = torch.stack(decoder_output.scores)


        # 여기 batch_decode에서 id -> token이 이루어짐
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


        
        output["logits"] = logit_stack # 두 번째 방식으로 구한 거 
        # output["logits"] = torch.tensor(softmax_scores).unsqueeze(1) # 첫 번째 방식으로 구한 확률 

        # output["logits"] = decoder_output.scores
        output["prob"] = softmax_scores # 이건 기존 lm_head에서 나온 값으로
        output["model_pred_id"] = model_pred_ids

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

    # 아래는 내가 추가한 부분 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 이 부분은 train 할 때만 실행되고, test때는 실행 안됨 
            module.weight.data.normal_(mean=0.0, std=0.05)  
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.05) 
            if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    
