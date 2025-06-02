import random
from typing import List, Tuple

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import random
from PIL import Image
from arguments import DataArguments

import logging
logger = logging.getLogger(__name__)

def filter_narrow_bbx(group):
    # if x1 and x2 are too close, then it is a narrow bounding box
    # or if y1 and y2 are too close, then it is a narrow bounding box
    return (group['bounding_box'][2] - group['bounding_box'][0] > 10) and (group['bounding_box'][3] - group['bounding_box'][1] > 10)

def filter_too_edge_bbx(group):
    # if x1 to close to the left edge, then it is a narrow bounding box
    # or if x2 to close to the right edge, then it is a narrow bounding box
    # or if y1 to close to the top edge, then it is a narrow bounding box
    # or if y2 to close to the bottom edge, then it is a narrow bounding box
    image_size = group['image'].size
    return (group['bounding_box'][0] > 10) and (group['bounding_box'][2] < image_size[0] - 10) and (group['bounding_box'][1] > 10) and (group['bounding_box'][3] < image_size[1] - 10)


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data_wiki = load_dataset('MrLight/nq-visa-multi-na')['train'].remove_columns(['url', 'short_answer_type', 'long_answer', 'image_size'])
        self.train_data_fb = load_dataset('MrLight/fineweb-visa-multi-na')['train'].filter(filter_narrow_bbx, num_proc=16, batch_size=1).filter(filter_too_edge_bbx, num_proc=16, batch_size=1)
        self.train_data_pub = load_dataset('MrLight/publaynet-visa-multi-na')['train']
        self.train_data_wiki_p1 = self.train_data_wiki.filter(lambda x: x['long_answer_type'] == 'p' and x['bounding_box'][1] < 980, num_proc=16, batch_size=1).shuffle().select(range(10000))
        self.train_data_wiki_p2 = self.train_data_wiki.filter(lambda x: x['long_answer_type'] == 'p' and x['bounding_box'][1] > 980, num_proc=16, batch_size=1).shuffle().select(range(10000))
        self.train_data_wiki_other = self.train_data_wiki.filter(lambda x: x['long_answer_type'] != 'p', num_proc=16, batch_size=1).select(range(10000))
        self.train_data_pub_text = self.train_data_pub.filter(lambda x: x['long_answer_type'] == 'text', num_proc=16, batch_size=1).shuffle().select(range(20000))
        self.train_data_pub_non_text = self.train_data_pub.filter(lambda x: x['long_answer_type'] != 'text', num_proc=16, batch_size=1).select(range(20000))
       
        self.data_to_use = []
        if 'wiki' in self.data_args.dataset_name:
            self.data_to_use.append(self.train_data_wiki_p1)
            self.data_to_use.append(self.train_data_wiki_p2)
            self.data_to_use.append(self.train_data_wiki_other)
        if 'fb' in self.data_args.dataset_name:
            self.data_to_use.append(self.train_data_fb)
        if 'pub' in self.data_args.dataset_name:
            self.data_to_use.append(self.train_data_pub_text)
            self.data_to_use.append(self.train_data_pub_non_text)
        self.train_data = concatenate_datasets(self.data_to_use)
        
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)
        

    def __getitem__(self, item) -> Tuple[str, str, str]:
        group = self.train_data[item]
        image = group['image']
        short_answer = group['short_answer']
        question = group['question']
        bbx = group['bounding_box']
        image_size = image.size
        
        target_width = 700
        scale = image_size[0] / target_width


        image = image.resize((int(image_size[0] / scale), int(image_size[1] / scale)))
        image_size = image.size
        
        # Scale the bounding box 
        bbx = [int( bbx[0] / scale), int(bbx[1] / scale), 
            int(bbx[2] / scale), int(bbx[3] / scale)]
        
        if self.data_args.augmentation:
            # Create a random crop that contains the bounding box
            crop_x1 = random.randint(0, max(0, bbx[0] - 5))
            crop_y1 = random.randint(0, max(0, bbx[1] - 5))
            crop_x2 = random.randint(min(bbx[2] + 5, image_size[0]), image_size[0])
            crop_y2 = random.randint(min(bbx[3] + 5, image_size[1]), image_size[1])

            # Crop the image using PIL
            cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # 10% of the time, we don't crop the image
            if random.random() < 0.1:
                cropped_image = image
                crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, image_size[0], image_size[1]
            
            # Update the bounding box based on the new crop
            bbx = [bbx[0] - crop_x1, bbx[1] - crop_y1, bbx[2] - crop_x1, bbx[3] - crop_y1]
            image = cropped_image


        # add noise to the bounding box by moving it by a few pixels
        bbx[0] += random.randint(-3, 3)
        bbx[1] += random.randint(-3, 3)
        bbx[2] += random.randint(-3, 3)
        bbx[3] += random.randint(-3, 3)

        if self.data_args.round_bbox:
            # round the bounding box value to times of 5
            bbx = [int(x / 5) * 5 for x in bbx]

        if self.data_args.normalize_bbx:
            # normalize the bounding box to [0, 1]
            bbx[0] /= image_size[0]
            bbx[1] /= image_size[1]
            bbx[2] /= image_size[0]
            bbx[3] /= image_size[1]
            bbx[0] = str(bbx[0])[:5]
            bbx[1] = str(bbx[1])[:5]
            bbx[2] = str(bbx[2])[:5]
            bbx[3] = str(bbx[3])[:5]

        bbx_text = '({}, {}), ({}, {})'.format(*bbx)

        if self.data_args.no_bbox:
            target_text = f'Answer: {short_answer}'
        elif self.data_args.no_ans:
            target_text = f'Bounding Box: {bbx_text}'
        else:
            target_text = f'Answer: {short_answer}\nBounding Box: {bbx_text}'

        return image, question, target_text
