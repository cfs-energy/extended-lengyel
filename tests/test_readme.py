import pytest


def readme_code_blocks():
    """Return the contents of every ```python block in the README."""
    from pathlib import Path

    readme_text = (Path(__file__).parents[1]/"README.md").read_text().splitlines()

    blocks, block = [], None
    for line in readme_text:
        if block is None:
            if line.strip() == "```python":
                block = []
        elif line.strip() == "```":
            blocks.append("\n".join(block))
            block = None
        else:
            block.append(line)

    assert blocks, "No ```python blocks found in the README."
    return blocks


@pytest.mark.parametrize("block_number", range(len(readme_code_blocks())))
def test_readme(block_number):
    """Test the Python text from the README file."""
    exec(readme_code_blocks()[block_number])
