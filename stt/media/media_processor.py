import logging
from pathlib import Path

from ..data_models.parameters import MediaParameters
from .core.downsample_and_convert_to_mono import downsample_and_convert_to_mono

logger = logging.getLogger(__name__)


class MediaProcessor:
    def __init__(self, parameters: MediaParameters):
        self.parameters = parameters

    def downsample_and_convert_to_mono(self, input_path: Path, output_path: Path) -> None:
        downsample_and_convert_to_mono(input_path, output_path, self.parameters.sample_rate)
