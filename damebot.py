import math

import numpy as np
import torch
import torch.nn.functional
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("model/ernie-1.0")
model = AutoModelForMaskedLM.from_pretrained("model/ernie-1.0")

# reference: https://github.com/Ledenel/nlp-zh/blob/main/src/app.py
def fill_mask(text, banned_words=(), allowed_words=(), unique=False, top_k=64, soft_unique=False, top_rate=1,
              debug=False):
    # banned_words = list(banned_words) + extra_ban()
    banned_words = list(banned_words)
    filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
    special_ids = tokenizer.convert_tokens_to_ids(list(tokenizer.special_tokens_map.values()))
    ent = []
    i = 1
    mask_str = tokenizer.special_tokens_map["mask_token"]
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_str)
    seq_id = None
    with torch.no_grad():
        while "[MASK]" in text:
            filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
            print("processing", text)
            tokenized_batch = tokenizer([text])
            logits = model(**tokenized_batch.convert_to_tensors("pt"))
            logits = logits.logits[0]  # pick first item, only one

            """
            temperature = torch.ones((logits.shape[0], 1))
            thr = math.log(0.8)
            def temperature_equation(temp):
                # temp = torch.nn.functional.sigmoid(temp)
                logits_mod = temp * logits
                softmaxed = torch.nn.functional.log_softmax(logits_mod, 1)
                max_item = softmaxed.max(axis=1).values
                delta = torch.clip(max_item - thr, 0)
                return delta.sum()  
            temp = torch_solve(
                temperature_equation,
                temperature
            )
            """
            thr = math.log(0.99)
            temp = torch.clamp(torch.nn.functional.log_softmax(logits, 1).max(axis=1).values / thr, min=0, max=1)
            temp = temp.unsqueeze(-1)
            print(temp.flatten())
            logits = logits * temp

            neg_inf = logits.min() - 10000000  # margin
            print("neg inf", neg_inf)
            mask_location_pt = tokenized_batch.convert_to_tensors("pt").input_ids[0] == mask_token_id
            logits[:, filter_ids] = neg_inf  # remove banned words
            logits[:, special_ids] = neg_inf  # remove special tokens
            # FIXME: can't remove here, since softmaxed result is equal-probability.
            # logits[~mask_location_pt, :] = neg_inf # remove un-masked words
            topk = torch.topk(logits, k=top_k, dim=1)
            topk_i_ind = topk.indices.clone()
            topk_i_ind[:, :] = torch.arange(topk_i_ind.shape[0]).unsqueeze(-1)
            topk_coo = torch.stack([topk_i_ind.view(-1), topk.indices.view(-1)])
            logits = torch.sparse_coo_tensor(topk_coo, topk.values.view(-1), logits.shape)  # clip logits to top-k
            # logits = sparse_mul(temp, logits)
            logits = torch.sparse.log_softmax(logits, 1)  # softmax it

            decreased_by_word_pow = torch.bincount(tokenized_batch.convert_to_tensors("pt").input_ids[0],
                                                   minlength=logits.shape[-1]).unsqueeze(0) + 1

            if soft_unique:
                if seq_id is not None:
                    seq_pow = torch.abs(torch.arange(logits.shape[0]) - seq_id).unsqueeze(-1)
                    decreased_by_word_pow = seq_pow @ decreased_by_word_pow
                logits = sparse_mul(decreased_by_word_pow,
                                    logits)  # make word with n-count P(word) ^ n, which is equalivant to n * log(P(word))
                logits = torch.sparse.log_softmax(logits, 1)  # re-softmax the logits

            ent_index_sr = pd.Series(index=list(logits.coalesce().indices().detach().numpy()),
                                     data=logits.coalesce().values().detach().numpy())

            mask_location_pt_where, *_ = torch.where(mask_location_pt)
            ent_index_sr_is_mask = pd.Series(ent_index_sr.index.get_level_values(0)).isin(
                mask_location_pt_where.detach().numpy()).values
            ent_index_sr = ent_index_sr[ent_index_sr_is_mask]
            top_k_val = ent_index_sr.sort_values(ascending=False)
            if top_rate < 0:
                exp_item = np.exp(top_k_val)
                exp_item_rate = exp_item / exp_item.max()
                exp_item_mask = exp_item_rate >= 1 - top_rate
                print("top_rate from", len(exp_item_mask), "to", exp_item_mask.sum())
                top_k_val = top_k_val[exp_item_mask]

            top_k_val = top_k_val[:top_k]
            top_k_val_item = top_k_val.sample(n=1, weights=np.exp(top_k_val))
            top_k_val_item = list(top_k_val_item.to_dict().items())[0]
            idx_pack, log_p = top_k_val_item
            seq_id, word_id = idx_pack
            word_text = tokenizer.convert_ids_to_tokens(word_id)
            seq_ids_origin = tokenized_batch[0].ids.copy()
            seq_ids_origin[seq_id] = word_id  # replace word with generated, before seq id changed
            seq_ids_origin = seq_ids_origin[1:-1]  # remove [CLS] and [SEP]
            seq_texts = tokenizer.convert_ids_to_tokens(seq_ids_origin)

            ent.append((
                seq_id,
                i,
                word_text,
                float(np.exp(log_p)))
            )
            if unique:
                banned_words.append(word_text)
            text = "".join(seq_texts)
            # text[min_item.coords["seq"]] = min_item.coords["word"]
            # text = "".join(str(x) for x in text.data)
            i += 1
    ent.sort()
    return text, " ".join('{}{}{:.3}'.format(*x[1:]) for x in ent) if debug else "——damebot"

def fill_mask_minimum(text, banned_words=(), allowed_words=(), unique=False, top_k=64, soft_unique=False, top_rate=1,
              debug=False):
    banned_words = list(banned_words)
    # filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
    special_ids = tokenizer.convert_tokens_to_ids(list(tokenizer.special_tokens_map.values()))
    ent = []
    i = 1
    mask_str = tokenizer.special_tokens_map["mask_token"]
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_str)
    seq_id = None
    with torch.no_grad():
        while "[MASK]" in text:
            filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
            print("processing", text)
            tokenized_batch = tokenizer([text])
            logits = model(**tokenized_batch.convert_to_tensors("pt"))
            logits = logits.logits[0]  # pick first item, only one

            thr = math.log(0.99)
            temp = torch.clamp(torch.nn.functional.log_softmax(logits, 1).max(axis=1).values / thr, min=0, max=1)
            temp = temp.unsqueeze(-1)
            print(temp.flatten())
            logits = logits * temp

            neg_inf = logits.min() - 10000000  # margin
            print("neg inf", neg_inf)
            mask_location_pt = tokenized_batch.convert_to_tensors("pt").input_ids[0] == mask_token_id
            logits[:, filter_ids] = neg_inf  # remove banned words
            logits[:, special_ids] = neg_inf  # remove special tokens
            # FIXME: can't remove here, since softmaxed result is equal-probability.
            # logits[~mask_location_pt, :] = neg_inf # remove un-masked words
            topk = torch.topk(logits, k=top_k, dim=1)
            topk_i_ind = topk.indices.clone()
            topk_i_ind[:, :] = torch.arange(topk_i_ind.shape[0]).unsqueeze(-1)
            topk_coo = torch.stack([topk_i_ind.view(-1), topk.indices.view(-1)])
            logits = torch.sparse_coo_tensor(topk_coo, topk.values.view(-1), logits.shape)  # clip logits to top-k
            # logits = sparse_mul(temp, logits)
            logits = torch.sparse.log_softmax(logits, 1)  # softmax it

            decreased_by_word_pow = torch.bincount(tokenized_batch.convert_to_tensors("pt").input_ids[0],
                                                   minlength=logits.shape[-1]).unsqueeze(0) + 1

            if soft_unique:
                if seq_id is not None:
                    seq_pow = torch.abs(torch.arange(logits.shape[0]) - seq_id).unsqueeze(-1)
                    decreased_by_word_pow = seq_pow @ decreased_by_word_pow
                logits = sparse_mul(decreased_by_word_pow,
                                    logits)  # make word with n-count P(word) ^ n, which is equalivant to n * log(P(word))
                logits = torch.sparse.log_softmax(logits, 1)  # re-softmax the logits

            ent_index_sr = pd.Series(index=list(logits.coalesce().indices().detach().numpy()),
                                     data=logits.coalesce().values().detach().numpy())

            mask_location_pt_where, *_ = torch.where(mask_location_pt)
            ent_index_sr_is_mask = pd.Series(ent_index_sr.index.get_level_values(0)).isin(
                mask_location_pt_where.detach().numpy()).values
            ent_index_sr = ent_index_sr[ent_index_sr_is_mask]
            top_k_val = ent_index_sr.sort_values(ascending=False)
            if top_rate < 0:
                exp_item = np.exp(top_k_val)
                exp_item_rate = exp_item / exp_item.max()
                exp_item_mask = exp_item_rate >= 1 - top_rate
                print("top_rate from", len(exp_item_mask), "to", exp_item_mask.sum())
                top_k_val = top_k_val[exp_item_mask]

            top_k_val = top_k_val[:top_k]
            top_k_val_item = top_k_val.sample(n=1, weights=np.exp(top_k_val))
            top_k_val_item = list(top_k_val_item.to_dict().items())[0]
            idx_pack, log_p = top_k_val_item
            seq_id, word_id = idx_pack
            word_text = tokenizer.convert_ids_to_tokens(word_id)
            seq_ids_origin = tokenized_batch[0].ids.copy()
            seq_ids_origin[seq_id] = word_id  # replace word with generated, before seq id changed
            seq_ids_origin = seq_ids_origin[1:-1]  # remove [CLS] and [SEP]
            seq_texts = tokenizer.convert_ids_to_tokens(seq_ids_origin)

            ent.append((
                seq_id,
                i,
                word_text,
                float(np.exp(log_p)))
            )
            if unique:
                banned_words.append(word_text)
            text = "".join(seq_texts)
            i += 1
    ent.sort()
    return text, " ".join('{}{}{:.3}'.format(*x[1:]) for x in ent) if debug else "——damebot"


# text = '[CLS]试看今日之域中，竟是谁家之天[MASK]。[SEP]'
text = '[CLS]试看今日之域中，竟是谁家之[MASK][MASK]。[SEP]'

filled_text, score = fill_mask_minimum(text, debug=True)
print("filled text: " + filled_text)
print("score: " + score)

filled_text, score = fill_mask_minimum(text, debug=False)
print("filled text: " + filled_text)
print("score: " + score)

filled_text, score = fill_mask_minimum(text)
print("filled text: " + filled_text)
print("score: " + score)