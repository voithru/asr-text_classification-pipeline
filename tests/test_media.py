from pathlib import Path
from tempfile import TemporaryDirectory

from stt.data_models.parameters import MediaParameters
from stt.media.media_processor import MediaProcessor


def test_media_processor():
    parameters = MediaParameters(sample_rate=16000)
    processor = MediaProcessor(parameters=parameters)

    with TemporaryDirectory() as tmp_dir:
        input_path = Path("tests/data/When_the_Weather_Is_Fine_12_4_stereo.wav")
        output_path = Path(tmp_dir) / "output.wav"

        processor.downsample_and_convert_to_mono(input_path, output_path)

        assert output_path.exists()
