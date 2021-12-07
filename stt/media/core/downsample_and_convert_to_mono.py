import logging
from pathlib import Path

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def downsample_and_convert_to_mono(input_path: Path, output_path: Path, sample_rate: int) -> None:
    output, rate = librosa.load(str(input_path), sr=sample_rate, mono=True)
    sf.write(str(output_path), output, rate)
