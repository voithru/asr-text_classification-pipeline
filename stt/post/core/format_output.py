import json
from pathlib import Path
from typing import List


def format_output(input_paths: List[Path], output_path: Path):
    input_paths.sort()
    with output_path.open("w") as output_file:
        result = dict()
        for input_path in input_paths:
            with input_path.open("r") as input_file:
                input_json = json.load(input_file)
                result[f"{input_path.stem}.wav"] = {"text": input_json["text"]}
        json.dump(result, output_file, ensure_ascii=False, indent=4)
