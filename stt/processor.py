import logging
from pathlib import Path
from typing import List

from .asr.asr_processor import ASRProcessor
from .data_models.parameters import ASRParameters, MediaParameters
from .media.media_processor import MediaProcessor
from .post.post_processor import PostProcessor

logger = logging.getLogger(__name__)


class Wav2vecProcessor:
    def __init__(self, media_params: MediaParameters, asr_params: ASRParameters):
        self.media_params = media_params
        self.asr_parameters = asr_params

    def process(self, input_dir: Path, output_path: Path):
        input_file_paths = list(input_dir.glob("*.wav"))
        intermediate_output_dir = output_path.parent.joinpath("intermediate")
        intermediate_output_dir.mkdir(parents=True)

        self.process_media(input_file_paths, intermediate_output_dir)
        self.process_asr(input_file_paths, intermediate_output_dir)
        self.postprocess(input_file_paths, intermediate_output_dir, output_path)

    def process_media(self, input_file_paths: List[Path], output_dir: Path):
        logger.info("Media processing START")
        media_processor = MediaProcessor(self.media_params)

        for input_file_path in input_file_paths:
            inter_output_dir = output_dir.joinpath(input_file_path.stem)
            inter_output_dir.mkdir()

            downsample_wav_path = inter_output_dir.joinpath("downsample.wav")
            media_processor.downsample_and_convert_to_mono(input_file_path, downsample_wav_path)

        logger.info("Media processing DONE")

    def process_asr(self, input_file_paths: List[Path], output_dir: Path):
        logger.info("ASR START")
        asr_processor = ASRProcessor(self.asr_parameters)

        for input_file_path in input_file_paths:
            inter_output_dir = output_dir.joinpath(input_file_path.stem)
            downsampled_audio_path = inter_output_dir.joinpath("downsample.wav")
            asr_path = inter_output_dir.joinpath(f"{input_file_path.stem}.json")
            asr_processor.speech_recognition(downsampled_audio_path, asr_path)

    def postprocess(
        self,
        input_file_paths: List[Path],
        intermediate_output_dir: Path,
        output_path: Path,
    ):  # pylint: disable=R0201
        logger.info("Postprocessing START")
        postprocessor = PostProcessor()
        subtitles_paths = [
            intermediate_output_dir.joinpath(f"{input_file_path.stem}/{input_file_path.stem}.json")
            for input_file_path in input_file_paths
        ]
        postprocessor.format_output(subtitles_paths, output_path)
        logger.info("Postprocessing DONE")
