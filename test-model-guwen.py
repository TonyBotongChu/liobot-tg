import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("model/guwenbert-base")
model = AutoModelForMaskedLM.from_pretrained("model/guwenbert-base")

# tokenizer = AutoTokenizer.from_pretrained("model/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("model/roberta-base")

# tokenizer = AutoTokenizer.from_pretrained("model/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("model/bert-base-uncased")

model.eval()
text = '[CLS][MASK]太元中，武陵人捕鱼为业。[SEP]'
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

exit()
