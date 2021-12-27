import logging
from pathlib import Path

from stt.parameters import ASRParameters, MediaParameters
from stt.processor import Wav2vecProcessor
from text_classification.predict import predict_text_class

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    wave_dir = Path("tests/data")
    stt_output_path = Path("./stt.json")
    output_path = Path("./predict.json")

    asr_param = ASRParameters(
        vocab_path=Path("tests/data/vocab.json"), checkpoint_path=Path("tests/data/checkpoint-700")
    )
    media_param = MediaParameters(sample_rate=16000)

    selected_processor = Wav2vecProcessor(media_param, asr_param)
    selected_processor.process(wave_dir, stt_output_path)
    predict_text_class(
        stt_output_path,
        output_path,
        checkpoint_dir=Path("tests/data/checkpoint-540"),
        config_path=Path("text_classification/koelectra-base-v3.json"),
    )


if __name__ == "__main__":
    main()
