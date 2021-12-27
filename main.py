import logging
from argparse import ArgumentParser
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


def main(args):
    wav_dir = args.wav_dir
    stt_output_path = args.stt_output_path
    output_path = args.output_path

    asr_param = ASRParameters(
        vocab_path=Path("tests/data/vocab.json"), checkpoint_path=args.wav2vec_checkpoint
    )
    media_param = MediaParameters(sample_rate=16000)

    selected_processor = Wav2vecProcessor(media_param, asr_param)
    selected_processor.process(wav_dir, stt_output_path)
    predict_text_class(
        stt_output_path,
        output_path,
        checkpoint_dir=args.electra_checkpoint,
        config_path=Path("text_classification/koelectra-base-v3.json"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wav_dir", type=Path)
    parser.add_argument("--stt_output_path", type=Path)
    parser.add_argument("--output_path", type=Path)

    parser.add_argument("--wav2vec_checkpoint", type=Path)
    parser.add_argument("--electra_checkpoint", type=Path)

    args = parser.parse_args()
    main(args)
