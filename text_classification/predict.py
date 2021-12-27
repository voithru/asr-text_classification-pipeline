import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from attrdict import AttrDict

from .predictor import GrandChallengeTextClassifier
from .text_processor import load_examples

logger = logging.getLogger(__name__)


def predict_text_class(
    input_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    config_path: Path,
):
    with config_path.open() as config_file:
        args = AttrDict(json.load(config_file))
        args.ckpt_dir = str(checkpoint_dir)

    predictor = GrandChallengeTextClassifier(args)
    test_dataset, file_name_list = load_examples(input_path, predictor.tokenizer, args)

    logger.info("TEXT CLASSIFICATION START")
    predictions = predictor.predict(test_dataset)
    results = convert_results_to_json(predictions, predictor.model.config.id2label, file_name_list)

    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=4)

    logger.info("TEXT CLASSIFICATION DONE")


def convert_results_to_json(predictions: np.ndarray, id2label: Dict, file_name_list: List):
    results = []
    for file_name, pred in zip(file_name_list, predictions):
        label = id2label[pred]
        results.append({"file_name": f"{file_name}", "class_code": label})

    results = sorted(results, key=lambda x: x["file_name"])

    return {"answer": results}
