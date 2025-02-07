from pathlib import Path

from grumpy.flow import Flow
from grumpy.types import Image, String

__version__ = Path(__file__).parent.joinpath("version.txt").read_text().strip()

__all__ = ["Flow", "__version__", "Image", "String"]
