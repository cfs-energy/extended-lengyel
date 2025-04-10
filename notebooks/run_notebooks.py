#!/usr/bin/env python3
import subprocess
from pathlib import Path

def execute_notebooks():
    """Executes all Jupyter notebooks in the script folder."""

    folder_path = Path(__file__).parent
    all_passed = True

    for filepath in sorted(folder_path.glob("*.ipynb")):
        print(f"Executing: {filepath}")
        
        try:
            subprocess.run(
                [
                    "jupyter", "nbconvert",
                    "--ClearMetadataPreprocessor.enabled=True",
                    "--execute",
                    "--to", "notebook",
                    "--inplace",
                    str(filepath)
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Successfully executed: {filepath}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {filepath}:")
            print(e.stderr)
            all_passed = False
    
    assert all_passed
        
if __name__ == "__main__":
    execute_notebooks()