from pathlib import Path
from typing import List

from .core.format_output import format_output


class PostProcessor:
    def __init__(self):
        pass

    def format_output(self, input_paths: List[Path], output_path: Path):
        format_output(input_paths, output_path)
