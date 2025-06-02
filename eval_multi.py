from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from datasets import load_dataset, concatenate_datasets
import argparse
import copy
import json
import logging
import re
import unicodedata
from tqdm import tqdm
import numpy as np

import regex

logger = logging.getLogger(__name__)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answers(text, answers, tokenizer, regex=False):
    text = _normalize(text)
    if regex:
        for ans in answers:
            ans = _normalize(ans)
            if regex_match(text, ans):
                return True
    else:
        text = tokenizer.tokenize(text).words(uncased=True)
        for ans in answers:
            ans = _normalize(ans)
            ans = tokenizer.tokenize(ans).words(uncased=True)
            for i in range(0, len(text) - len(ans) + 1):
                if ans == text[i: i + len(ans)]:
                    return True
    return False

tokenizer = SimpleTokenizer()

import argparse
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='')
parser.add_argument('--setting', type=str, required=True, help='')
parser.add_argument('--model', type=str, required=True, help='')
args = parser.parse_args()

if args.setting in ['n', 's', 'f']:
    train_corpus = load_dataset('MrLight/nq-visa-multi-na', split='train')
    test_corpus = load_dataset('MrLight/nq-visa-multi-na', split='test')
    corpus = concatenate_datasets([train_corpus, test_corpus])
if args.setting in ['t', 'nt']:
    train_corpus = load_dataset('MrLight/publaynet-visa-multi-na', split='train')
    test_corpus = load_dataset('MrLight/publaynet-visa-multi-na', split='test')
    corpus = concatenate_datasets([train_corpus, test_corpus])

docid2idx = {}
for idx, docid in enumerate(corpus['id']):
    docid2idx[str(docid)] = idx

print(docid2idx)


if args.setting == 'n':
    nq_dev = load_dataset("MrLight/nq-visa-multi-na")['test'].filter(lambda x: x['long_answer_type'] != 'p', num_proc=16, batch_size=1).filter(lambda x: x['pos_idx'] != -1, num_proc=16, batch_size=1)
if args.setting == 'f':
    nq_dev = load_dataset("MrLight/nq-visa-multi-na")['test'].filter(lambda x: x['long_answer_type'] == 'p' and x['bounding_box'][1] < 980, num_proc=16, batch_size=1).filter(lambda x: x['pos_idx'] != -1, num_proc=16, batch_size=1)
if args.setting == 's':
    nq_dev = load_dataset("MrLight/nq-visa-multi-na")['test'].filter(lambda x: x['long_answer_type'] == 'p' and x['bounding_box'][1] >= 980, num_proc=16, batch_size=1).filter(lambda x: x['pos_idx'] != -1, num_proc=16, batch_size=1)
if args.setting == 't':
    nq_dev = load_dataset("MrLight/publaynet-visa-multi-na")['test'].filter(lambda x: x['long_answer_type'] == 'text', num_proc=16, batch_size=1).filter(lambda x: len(x['candidates']) > 0, num_proc=16, batch_size=1).filter(lambda x: x['pos_idx'] != -1, num_proc=16, batch_size=1)
if args.setting == 'nt':
    nq_dev = load_dataset("MrLight/publaynet-visa-multi-na")['test'].filter(lambda x: x['long_answer_type'] != 'text', num_proc=16, batch_size=1).filter(lambda x: len(x['candidates']) > 0, num_proc=16, batch_size=1).filter(lambda x: x['pos_idx'] != -1, num_proc=16, batch_size=1)


# nq_dev = load_dataset("MrLight/nq-visa-multi-na", token='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')['train'].shuffle(55).select(range(200))
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    f"qwen_multi/{args.model}",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def has_overlap(a, b):
    # Extract coordinates
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Check if there is no overlap
    if ax1 >= bx2 or bx1 >= ax2:  # One box is to the left of the other
        return False
    if ay1 >= by2 or by1 >= ay2:  # One box is above the other
        return False

    # Otherwise, there is an overlap
    return True

def enough_iou(a, b, threshold=0.5):
    # Extract coordinates
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Calculate the area of the boxes
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    # Calculate the coordinates of the intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # Calculate the area of the intersection
    i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Calculate the area of the union
    u_area = a_area + b_area - i_area

    # Calculate the IoU
    iou = i_area / u_area

    # Check if the IoU is greater than the threshold
    return iou >= threshold



results = []
answer_results = []
with open(f'{args.model}_{args.setting}_multi_output_ha.jsonl', 'w') as f:
    for num in tqdm(range(len(nq_dev))):
        group = nq_dev[num]
        candidates = group['candidates']
        pos_idx = group['pos_idx']
        short_answer = group['short_answer']
        question = group['question']
        bbx = group['bounding_box']
        image_size = group['image'].size
        target_width = 700
        scale = image_size[0] / 700
        bbx = [int( bbx[0] / scale), int(bbx[1] / scale), 
                int(bbx[2] / scale), int(bbx[3] / scale)]
        images = []
        for idx, candidate in enumerate(candidates):
            image = corpus[docid2idx[candidate]]['image']
            image_size = image.size
            scale = image_size[0] / 700
            image = image.resize((int(image_size[0] / scale), int(image_size[1] / scale)))
            images.append(image)


        messages = [
            {
                'role': 'system',
                'content': [
                    {'type': 'text', 'text': 'Given a document image, your task is to answer the question and provide the bounding box of the answer.'}
                ]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': images[0]},
                    {'type': 'text', 'text': f'Image Size: {images[0].size}\n'},
                    {'type': 'image', 'image': images[1]},
                    {'type': 'text', 'text': f'Image Size: {images[1].size}\n'},
                    {'type': 'image', 'image': images[2]},
                    {'type': 'text', 'text': f'Image Size: {images[2].size}\n'},
                    {'type': 'text', 'text': f'Question: {question}'}
                ]
            },
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, top_k=None, top_p=None)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )       

        ground_truth = short_answer

        # print(output_text[0])
        # try:
        answer = output_text[0]
        if "Bounding Box" not in answer:
            if pos_idx == -1 and "No Answer" in answer:
                results.append(True)
                answer_results.append(True)
            else:
                results.append(False)
                answer_results.append(False)
        else:
            predic_doc = output_text[0].split("Target Document: ")[1].split("\n")[0]
            predic_doc = int(predic_doc) - 1
            if predic_doc != pos_idx:
                results.append(False)
            else:
                bounding_box = output_text[0].split("Bounding Box: ")[1]
                bounding_box = bounding_box.replace("(", "").replace(")", "").split(",")
                bounding_box = [int(coord) for coord in bounding_box]
                results.append(enough_iou(bbx, bounding_box))
            answer = output_text[0].split("Target Document: ")[0].replace('Answer:', '').strip()
            if (has_answers(ground_truth, [answer], tokenizer) or has_answers(answer, [ground_truth], tokenizer)) and abs(len(answer) - len(ground_truth))<20:
                answer_results.append(True)
            else:
                answer_results.append(False)
        f.write(json.dumps(
            {
                'question': question,
                'answer': ground_truth,
                'output': output_text
            }
        ) + '\n')

        # except:
        #     results.append(False)
        #     answer_results.append(False)
        #     f.write(json.dumps(
        #         {
        #             'question': question,
        #             'answer': ground_truth,
        #             'output': ''
        #         }
        #     ) + '\n')
        print(sum(results) / len(results), sum(answer_results) / len(answer_results))
    print(sum(results) / len(results), sum(answer_results) / len(answer_results))
