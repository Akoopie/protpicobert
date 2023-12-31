import os
import pathlib

import sequence_utils as su
import token_utils as tu
import train

project_dir = pathlib.Path(__file__).parent.parent

data_dir = project_dir / "data"
records_file = data_dir / "lacc.fasta"
tokenizer_file = data_dir / "tokenizer.json"
token_dir = data_dir / "tokens"

records = su.parse_records(records_file)

#tokenizer = tu.train_tokenizer(records, str(tokenizer_file))

tokenizer  = tu.load_tokenizer(str(tokenizer_file))
records = {
    "input_ids": records
}
tu.tokenize_data(str(tokenizer_file), records, token_dir)

tokenized_dataset = tu.load_tokenized_data(token_dir)

train.train_model(str(tokenizer_file), tokenized_dataset)