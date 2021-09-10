import torch
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

class BotBackend:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained("model/" + model_name)
        self.model = AutoModelForMaskedLM.from_pretrained("model/" + model_name)

    def fill_mask(self, text):
        while self.tokenizer.mask_token in text:
            input = self.tokenizer.encode(text, return_tensors="pt")
            mask_token_indexes = torch.where(input == self.tokenizer.mask_token_id)[1]
            # mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
            mask_token_index = torch.reshape(random.choice(mask_token_indexes), (1,))
            token_logits = self.model(input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]
            # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            predict_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()[0]
            # predict_word = self.tokenizer.decode([predict_token])
            output = input.clone()
            # current_token = output[0][mask_token_index.tolist()[0]].tolist()
            output[0][mask_token_index.tolist()[0]] = predict_token
            text = self.tokenizer.convert_ids_to_tokens(output[0])
            # remove redundant [CLS] and [SEP]
            text = text[1:-1]
            text = ''.join(text)
        return text


if __name__ == "__main__":
    bot_backend = BotBackend("guwenbert-large")
    text = '[CLS]试看今日之域中，竟是谁家之[MASK][MASK]。[SEP]'
    text = bot_backend.fill_mask(text)
    print(text)