# %%
import os
import token_utils as tu
from transformers import BertForMaskedLM
from transformers import pipeline
import transformers as tr

project_dir = os.getcwd()
data_dir = project_dir + "/data"
records_file = data_dir + "/tiny_lacc.fasta"
tokenizer_file = data_dir + "/tokenizer.json"
token_dir = data_dir + "/tokens"
tokenizer = tr.PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, return_special_tokens_mask = True)
tokenizer.mask_token = "[MASK]"
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"
tokenizer.mask_token_id = 1
tokenizer.pad_token_id = 2
tokenizer.cls_token_id = 0
bert = BertForMaskedLM.from_pretrained("./data\checkpoint\checkpoint-10000")
pipe = pipeline("fill-mask", model=bert, tokenizer=tokenizer)
print(pipe("[MASK]"))
