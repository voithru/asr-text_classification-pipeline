from pathlib import Path
from tempfile import TemporaryDirectory

from stt.asr.asr_processor import ASRProcessor
from stt.parameters import ASRParameters


def test_media_processor():
    asr_params = ASRParameters(Path("tests/data/vocab.json"), Path("tests/data/checkpoint-700"))
    processor = ASRProcessor(asr_params)

    with TemporaryDirectory() as tmp_dir:
        input_path = Path("tests/data/When_the_Weather_Is_Fine_12_4_stereo.wav")
        output_path = Path(tmp_dir) / "output.wav"

        processor.speech_recognition(input_path, output_path)

        assert output_path.exists()
