from pathlib import Path
from tempfile import TemporaryDirectory

from stt.asr.asr_processor import ASRProcessor


def test_media_processor():
    processor = ASRProcessor(Path("tests/data/vocab.json"), Path("tests/data/checkpoint-700"))

    with TemporaryDirectory() as tmp_dir:
        input_path = Path("tests/data/When_the_Weather_Is_Fine_12_4_stereo.wav")
        output_path = Path(tmp_dir) / "output.wav"

        processor.speech_recognition(input_path, output_path)

        assert output_path.exists()
