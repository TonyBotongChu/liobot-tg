'''
source:
https://stackoverflow.com/a/54979815
https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model#comment104132471_54979815
https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model#comment101453193_54979815
'''

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("model/guwenbert-base")
# model = AutoModelForMaskedLM.from_pretrained("model/guwenbert-base")

# tokenizer = AutoTokenizer.from_pretrained("model/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("model/roberta-base")

tokenizer = AutoTokenizer.from_pretrained("model/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("model/bert-base-uncased")

model.eval()
text = '[CLS] I want to [MASK] the car because it is cheap . [SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the segments tensors.
segments_ids = [0] * len(tokenized_text)

masked_index = tokenized_text.index('[MASK]')

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_token)

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
