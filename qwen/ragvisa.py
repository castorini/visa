from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

from arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


class RagVisaModel(nn.Module):
    TRANSFORMER_CLS = Qwen2VLForConditionalGeneration

    def __init__(self, generator: PreTrainedModel):
        super().__init__()
        self.config = generator.config
        self.config.hidden_size = 3584 #1536 #3584
        self.hidden_size = 3584 # 1536 #3584
        self.generator = generator

    def forward(self, inputs):

        if self.training:
            cache_position = torch.arange(0, inputs['input_ids'].shape[0], device=inputs['input_ids'].device)
            labels = inputs['labels']
            inputs = self.generator.prepare_inputs_for_generation(**inputs, use_cache=True, cache_position=cache_position)
            outputs = self.generator(**inputs, return_dict=True, labels=labels)

        return outputs
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.generator.model.gradient_checkpointing_enable()

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, trust_remote_code=True)
        base_model.padding_side = "left"
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=['kqv', 'fc1', 'fc2', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], #model_args.lora_target_modules.split(','),
                    lora_dropout=model_args.lora_dropout,
                    init_lora_weights="gaussian",
                    use_dora=True,
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                generator=lora_model
            )
        else:
            model = cls(
                generator=base_model
            )
        return model

    def save(self, output_dir: str):
        self.generator.save_pretrained(output_dir)