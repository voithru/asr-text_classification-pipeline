import json
from pathlib import Path
from tempfile import TemporaryDirectory

from stt.post.post_processor import PostProcessor


def test_postprocess():
    processor = PostProcessor()

    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "output.json"
        input_paths = [Path("tests/data/When_the_Weather_Is_Fine_12_4_stereo.json")]
        processor.format_output(input_paths, output_path)

        with output_path.open() as output_file:
            output = json.load(output_file)

        assert "When_the_Weather_Is_Fine_12_4_stereo.wav" in output
        assert (
            output["When_the_Weather_Is_Fine_12_4_stereo.wav"]["text"][0]
            == "봐봐 자세히꼬아야고 그강프짓다 사 일년 나도 이때에 그치 이쁘지"
        )
