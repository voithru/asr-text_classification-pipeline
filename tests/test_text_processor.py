import json
from pathlib import Path

from attrdict import AttrDict
from transformers import ElectraTokenizer

from text_classification.text_processor import load_examples


def test_input_example():
    input_path = Path("tests/data/text_sample.json")
    tokenizer_path = Path("text_classification/text_classification_tokenizer")
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_path)

    config_path = Path("text_classification/koelectra-base-v3.json")
    with config_path.open() as config_file:
        args = AttrDict(json.load(config_file))
    dset, fname_list = load_examples(input_path, tokenizer, args)

    assert len(dset) == 2
    assert len(fname_list) == 2
    assert "t2_001.wav" in fname_list
    assert "t2_002.wav" in fname_list
