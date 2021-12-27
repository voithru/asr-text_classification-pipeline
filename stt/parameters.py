from dataclasses import dataclass
from pathlib import Path


@dataclass
class MediaParameters:
    sample_rate: int


@dataclass
class ASRParameters:
    vocab_path: Path
    checkpoint_path: Path
