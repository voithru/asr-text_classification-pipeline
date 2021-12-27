import copy
import json

import torch
from torch.utils.data import TensorDataset


class InputExample:
    """
    A single training/tests example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def seq_cls_convert_examples_to_features(examples, tokenizer, max_length):

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=None)
        features.append(feature)

    return features


class TextProcessor:
    def __init__(self):
        pass

    def get_labels(self):
        return ["020121", "02051", "020811", "020819", "000001"]

    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _create_examples(self, data):
        examples = []
        file_name_list = []
        for (i, file_name) in enumerate(data):
            text_a = data[file_name]["text"]
            examples.append(InputExample(guid=file_name, text_a=text_a, text_b=None, label=None))
            file_name_list.append(file_name)
        return examples, file_name_list

    def get_examples(self, input_path):
        return self._create_examples(self._read_file(input_path))


def load_examples(input_path, tokenizer, max_seq_len):
    processor = TextProcessor()

    examples, file_name_list = processor.get_examples(input_path)

    features = seq_cls_convert_examples_to_features(examples, tokenizer, max_length=max_seq_len)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([0] * len(features), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset, file_name_list
