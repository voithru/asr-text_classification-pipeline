import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from text_classification.predict import predict_text_class


def test_predict_text_class():
    input_path = Path("tests/data/text_sample.json")
    checkpoint_path = Path("tests/data/checkpoint-540")
    config_path = Path("text_classification/koelectra-base-v3.json")
    with NamedTemporaryFile(suffix=".json") as tmp_file:
        predict_text_class(input_path, Path(tmp_file.name), checkpoint_path, config_path)

        assert Path(tmp_file.name).exists()
        with Path(tmp_file.name).open() as output_file:
            output_data = json.load(output_file)
            assert len(output_data["answer"]) == 2
            assert output_data["answer"][0]["file_name"] == "t2_001.wav"
