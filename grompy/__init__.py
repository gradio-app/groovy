from pathlib import Path

from grompy.transpiler import transpile

__version__ = Path(__file__).parent.joinpath("version.txt").read_text().strip()

__all__ = ["__version__", "transpile"]
