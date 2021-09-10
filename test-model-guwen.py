import torch
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("model/guwenbert-large")
model = AutoModelForMaskedLM.from_pretrained("model/guwenbert-large")

# tokenizer = AutoTokenizer.from_pretrained("model/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("model/roberta-base")

# tokenizer = AutoTokenizer.from_pretrained("model/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("model/bert-base-uncased")

model.eval()
text = '[CLS]试看今日之域中，竟是谁家之天[MASK]。[SEP]'
# text = '[CLS][MASK]太元中，武陵人捕鱼为业。[SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# # Create the segments tensors.
# segments_ids = [0] * len(tokenized_text)
#
# masked_index = tokenized_text.index('[MASK]')
#
# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
#
# # Predict all tokens
# with torch.no_grad():
#     predictions = model(tokens_tensor, segments_tensors)
#
# predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
#
# print(predicted_token)

text = '[CLS]试看今日之域中，竟是谁家之[MASK][MASK]。[SEP]'
# text = '[CLS]试看今日之域中，竟是谁家之天[MASK]。[SEP]'
while "[MASK]" in text:
    input = tokenizer.encode(text, return_tensors="pt")
    mask_token_indexes = torch.where(input == tokenizer.mask_token_id)[1]
    # mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    mask_token_index = torch.reshape(random.choice(mask_token_indexes), (1,))
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # mask_token_logits =
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    predict_token = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()[0]
    # text = text.replace()
    predict_word = tokenizer.decode([predict_token])
    output = input.clone()
    current_token = output[0][mask_token_index.tolist()[0]].tolist()
    output[0][mask_token_index.tolist()[0]] = predict_token
    text = tokenizer.convert_ids_to_tokens(output[0])
    text = text[1:-1]
    text = ''.join(text)
    print(text)

# sequence = '[CLS]试看今日之域中，竟是谁家之天[MASK]。[SEP]'
#
# input = tokenizer.encode(sequence, return_tensors="pt")
# mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
#
# token_logits = model(input)[0]
# mask_token_logits = token_logits[0, mask_token_index, :]
#
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#
# for token in top_5_tokens:
#     print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
