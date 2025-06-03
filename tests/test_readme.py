def test_readme():
    """Test the Python text from the README file."""
    from pathlib import Path

    readme_text = (Path(__file__).parents[1]/"README.md").read_text().splitlines()

    start_line = readme_text.index("```python")
    stop_line = readme_text[start_line:].index("```")

    exec("\n".join(readme_text[start_line+1:start_line+stop_line]))
