import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

from arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin


    def __call__(self, features: List[Tuple[str, str, str]]):
        all_images = [f[0] for f in features]
        all_questions = [f[1] for f in features]
        all_targets = [f[2] for f in features]
        messages = []
        for image, question, target in zip(all_images, all_questions, all_targets):
            message = [
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': 'Given three document images, your task is to answer the question, tell which document is the most relevant document and locate the source of the answer via a bounding box.'}
                    ]
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image[0]},
                        {'type': 'text', 'text': f'Image Size: {image[0].size}\n'},
                        {'type': 'image', 'image': image[1]},
                        {'type': 'text', 'text': f'Image Size: {image[1].size}\n'},
                        {'type': 'image', 'image': image[2]},
                        {'type': 'text', 'text': f'Image Size: {image[2].size}\n'},
                        {'type': 'text', 'text': f'Question: {question}'}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': target}
                    ]
                }
            ]
            messages.append(message)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
        )
        # only add the labels for the assistant
        # there is only on assistant message per example so any token after the assistant token has label as itselt, rest are -100
        labels = []
        for input_ids in inputs['input_ids']:
            assistant_idx = input_ids.index(self.processor.tokenizer.encode("assistant")[0])
            label = [-100] * len(input_ids)
            label[assistant_idx + 1 :] = input_ids[assistant_idx + 1 :]
            labels.append(label)
        
        # convert everything to tensors
        inputs['labels'] = torch.tensor(labels)
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}

        return inputs
