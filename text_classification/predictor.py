import logging
import random

import numpy as np
import torch
from fastprogress.fastprogress import progress_bar
from torch.utils.data import DataLoader, SequentialSampler
from transformers import ElectraForSequenceClassification, ElectraTokenizer

logger = logging.getLogger(__name__)


class GrandChallengeTextClassifier:
    def __init__(self, args):
        self.args = args

        self.tokenizer = ElectraTokenizer.from_pretrained(
            self.args.tokenizer_dir, do_lower_case=self.args.do_lower_case
        )

        logger.info("Predict text class the following checkpoints: %s", self.args.ckpt_dir)

        self.args.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"

        self.model = ElectraForSequenceClassification.from_pretrained(self.args.ckpt_dir)
        self.model.to(self.args.device)
        self.model.eval()

    def predict(self, test_dataset):
        sampler = SequentialSampler(test_dataset)
        dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=self.args.eval_batch_size)

        logger.info("Num examples = {}".format(len(test_dataset)))

        preds = None

        for batch in progress_bar(dataloader):
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                outputs = self.model(**inputs)
                logits = outputs[1]

            preds = logits.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)

        return preds
