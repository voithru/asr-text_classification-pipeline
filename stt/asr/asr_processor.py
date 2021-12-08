import json
from pathlib import Path

import librosa
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


class ASRProcessor:
    def __init__(self, vocab_path: Path, checkpoint_path: Path):
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor, tokenizer=self.tokenizer
        )

    def speech_recognition(self, audio_path: Path, output_path: Path):
        # audio, sample_rate = sf.read(audio_path, dtype=np.single, always_2d=True)
        # audio = audio.mean(axis=1)
        with torch.no_grad():
            audio, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            logits = self.model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = self.processor.batch_decode(predicted_ids)

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump({"text": transcription}, output_file, ensure_ascii=False)
