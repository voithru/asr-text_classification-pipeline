import json
from pathlib import Path
from tempfile import TemporaryDirectory

from stt.data_models.parameters import ASRParameters, MediaParameters
from stt.processor import Wav2vecProcessor


def test_stt():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir
        asr_param = ASRParameters(
            vocab_path=Path("tests/data/vocab.json"),
            checkpoint_path=Path("tests/data/checkpoint-700"),
        )
        media_param = MediaParameters(sample_rate=16000)

        wave_dir = Path("tests/data")
        stt_output_path = Path(tmp_dir) / "output.json"
        selected_processor = Wav2vecProcessor(media_param, asr_param)
        selected_processor.process(wave_dir, stt_output_path)

        with stt_output_path.open() as output_file:
            output = json.load(output_file)

        assert stt_output_path.exists()
        assert "When_the_Weather_Is_Fine_12_4_stereo.wav" in output
        assert (
            output["When_the_Weather_Is_Fine_12_4_stereo.wav"]["text"][0]
            == "봐봐 자세히꼬아야고 그강프짓다 사 일년 나도 이때에 그치 이쁘지"
        )
