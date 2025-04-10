"""Define relative paths, for finding data-files."""

from pathlib import Path

module_directory = Path(__file__).parent
library_directory = module_directory.parent
# Default radas directory
radas_dir = library_directory / "radas_dir"
notebook_dir = library_directory / "notebooks"

output_dir = library_directory / "output"
output_dir.mkdir(exist_ok=True)
