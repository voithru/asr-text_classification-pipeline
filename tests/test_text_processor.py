import json
from pathlib import Path

from attrdict import AttrDict
from transformers import ElectraTokenizer

from text_classification.text_processor import InputExample, InputFeatures, load_examples


def test_load_examples():
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


def test_input_example():
    input_example = InputExample(0, "HI", "TEST", 0)

    assert (
        input_example.__repr__()
        == '{\n  "guid": 0,\n  "label": 0,\n  "text_a": "HI",\n  "text_b": "TEST"\n}\n'
    )


def test_input_features():
    input_features = InputFeatures(0, 0, 0, 0)

    assert (
        input_features.__repr__()
        == '{\n  "attention_mask": 0,\n  "input_ids": 0,\n  "label": 0,\n  "token_type_ids": 0\n}\n'
    )
