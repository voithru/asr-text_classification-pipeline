from pathlib import Path

from transformers import ElectraTokenizer

from text_classification.text_processor import load_examples


def test_input_example():
    input_path = Path("tests/data/text_sample.json")
    tokenizer_path = Path("text_classification/text_classification_tokenizer")
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_path)
    dset, fname_list = load_examples(input_path, tokenizer, 100)

    assert len(dset) == 2
    assert len(fname_list) == 2
    assert "t2_001.wav" in fname_list
    assert "t2_002.wav" in fname_list
