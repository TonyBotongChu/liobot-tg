import numpy
import torch
import random

from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline

import logging

class BotBackend:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained("model/" + model_name)
        self.model = AutoModelForMaskedLM.from_pretrained("model/" + model_name)
        self.fm_pipeline = FillMaskPipeline(model=self.model, tokenizer=self.tokenizer)

    def fill_template(self, template, word):
        masked_text = ""
        masked_text += self.tokenizer.cls_token
        word_not_filled = word is not None
        for element in template:
            if isinstance(element, str):
                masked_text += element
            else:
                if word_not_filled:
                    masked_text += word
                    word_not_filled = False
                    continue
                min_len, max_len = element
                len = random.randint(min_len, max_len)
                masked_text += self.tokenizer.mask_token * len
        masked_text += self.tokenizer.sep_token
        return masked_text

    # TODO: add banned or excluded words here
    def fill_mask(self, text):
        while self.tokenizer.mask_token in text:
            result = self.fm_pipeline(text)
            if text.count(self.tokenizer.mask_token) == 1:
                text = result[0]['sequence']
            else:
                # the first index represents which <mask> to be filled
                # and the second index represents the candidate number, sort by score
                mask_to_fill = random.choice(result)
                scores = torch.zeros(len(mask_to_fill))
                for i in range(0, len(mask_to_fill)):
                    scores[i] = mask_to_fill[i]['score']
                s = nn.Softmax(dim=0)
                probabilities = s(scores)

                # numpy bug: raise 'ValueError: probabilities do not sum to 1' even if the sum is 1
                probabilities = numpy.asarray(probabilities).astype('float64')
                probabilities = probabilities / numpy.sum(probabilities)

                print(probabilities)
                text = numpy.random.choice(mask_to_fill, p=probabilities)['sequence']
        text = text.replace(' ', '')
        return text

if __name__ == "__main__":
    # bot_backend = BotBackend("guwenbert-large")
    bot_backend = BotBackend("ernie-1.0")
    text = '[CLS]试看今日之域中，竟是谁[MASK]之[MASK][MASK]。[SEP]'
    text = bot_backend.fill_mask(text)
    print(text)
    template = ["您天天都在",(2,5),"，您完全不",(2,3),"的是吗？"]
    text = bot_backend.fill_template(template, "摸鱼")
    print(text)
    text = bot_backend.fill_mask(text)
    print(text)
