import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs 
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import types
from typing import Tuple, Optional, Unpack, List, Literal, Callable, Dict, Union
from lightning import LightningModule
from litgpt.utils import chunked_cross_entropy
from functools import partial
from enum import Enum, auto

from .layer0_llama import LlamaForCausalLMFirstLayer



def mod_parallel_attention(model: torch.nn.Module):
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states_layer_norm = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states_attn, self_attn_weights = self.self_attn(
            hidden_states=hidden_states_layer_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states_mlp = self.mlp(hidden_states_layer_norm)
        
        result_hidden_states = residual + hidden_states_attn + hidden_states_mlp

        outputs = (result_hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs



    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):            
            module.forward = types.MethodType(new_forward, module)
    
    return model


def _mod_knn_as_lm_head(model: torch.nn.Module, inv_type: Literal['gaussian', 'neg_local', 'neg_local_global']='gaussian'):
    class KnnLmHead(torch.nn.Module):
        def __init__(self, emb_mat: torch.Tensor, inv_type: Literal['gaussian', 'neg_local', 'neg_local_global']='gaussian'):
            super().__init__()
            self.emb_mat = emb_mat
            self.inv_type = inv_type

            if inv_type == 'gaussian':
                self.sigmas = torch.nn.parameter.Parameter(
                    torch.ones((1, emb_mat.shape[0]), dtype=emb_mat.dtype, device=emb_mat.device)
                )
            elif inv_type == 'neg_local':
                self.temp = torch.nn.Linear(self.emb_mat.shape[1], 1)
            elif inv_type == 'neg_local_global':
                self.sigmas = torch.nn.parameter.Parameter(
                    torch.ones((1, emb_mat.shape[0]), dtype=emb_mat.dtype, device=emb_mat.device)
                )
                self.temp = torch.nn.Linear(self.emb_mat.shape[1], 1)


        def forward(self, hidden_state: torch.Tensor):
            flatten_hidden_state = hidden_state.view(-1, hidden_state.shape[-1])

            knn_logit = torch.cdist(
                flatten_hidden_state,
                self.emb_mat,
                # compute_mode='use_mm_for_euclid_dist'
            )
            if self.inv_type == 'gaussian':
                knn_logit = torch.exp(-knn_logit ** 2 / (2 * self.sigmas ** 2))
                knn_logit = knn_logit / torch.sum(knn_logit, dim=-1, keepdim=True)
            elif self.inv_type == "neg_local":
                curr_temp =  self.temp(flatten_hidden_state)
                knn_logit = - knn_logit * curr_temp
            elif self.inv_type == "neg_local_global":
                curr_temp =  self.temp(flatten_hidden_state)
                knn_logit = - knn_logit * self.sigmas * curr_temp
            else:
                raise RuntimeError(f'Catch inv_type: {self.inv_type}')

            knn_logit = knn_logit.view(hidden_state.shape[:-1] + (-1,))

            return knn_logit

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):            
            embb = module

    emb_weight = embb.weight

    for name, module in model.named_modules():
        if isinstance(module, LlamaForCausalLM):            
            module.set_output_embeddings(KnnLmHead(emb_weight, inv_type))
    
    return model


mod_knn_as_lm_head_gaussian: Callable = partial(_mod_knn_as_lm_head, inv_type='gaussian')
mod_knn_as_lm_head_neg_local: Callable = partial(_mod_knn_as_lm_head, inv_type='neg_local')
mod_knn_as_lm_head_neg_local_global: Callable = partial(_mod_knn_as_lm_head, inv_type='neg_local_global')


MODS = {
    "parallel_attention": mod_parallel_attention,
    "knn_as_lm_head_gaussian": mod_knn_as_lm_head_gaussian,
    "knn_as_lm_head_neg_local": mod_knn_as_lm_head_neg_local,
    "knn_as_lm_head_neg_local_global": mod_knn_as_lm_head_neg_local_global
}


LLAMA_LM_CLASSES = {
    "Base_Llama": LlamaForCausalLM,
    "First_Layer_Llama": LlamaForCausalLMFirstLayer
}


class LitLLM(LightningModule):
    def __init__(
            self, 
            llama_config: LlamaConfig, 
            llama_lm_class: str,
            mods: List[str] = [],
            use_compile: bool = True,
            total_steps: int=10, max_lr: float = 3e-4, warmup: float = 0.05,
        ):
        super().__init__()

        self.model = LLAMA_LM_CLASSES[llama_lm_class](llama_config)
        for mod in mods:
            self.model = MODS[mod](self.model)

        if use_compile:
            self.model = torch.compile(self.model)

        self.warmup = warmup
        self.total_steps = total_steps
        self.max_lr = max_lr

        self._consumed_tokens = 0
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        logits = outputs['logits'][..., :-1, :]
        targets = batch["input_ids"][..., 1:]

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss =  torch.nn.functional.cross_entropy(logits, targets, ignore_index=self.model.config.pad_token_id)

        # loss = chunked_cross_entropy(predict, target, chunk_size=256, ignore_index=self.model.config.pad_token_id)

        self.log("train/loss", loss, prog_bar=True)
        # self.log('train/steps', batch_idx // ACCUMULATED_BATCHES, prog_bar=True)

        self._consumed_tokens += batch['attention_mask'].sum().item()
        self.log('train/consumed_tokens', self._consumed_tokens)
        return loss
    
    def validation_step(self, batch):
        outputs = self.model(**batch)
        predict = outputs['logits'][..., :-1, :]
        target = batch["input_ids"][..., 1:]
        loss = chunked_cross_entropy(predict, target, chunk_size=128, ignore_index=self.model.config.pad_token_id)

        self.log("val/loss", loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.total_steps, pct_start=self.warmup, anneal_strategy='cos')
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
