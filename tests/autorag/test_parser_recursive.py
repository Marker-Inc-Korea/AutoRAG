from __future__ import annotations

import pathlib
import tempfile

import pandas as pd
from autorag.autorag.parser import Parser


HERE = pathlib.Path(__file__).resolve()
RESOURCE_DIR = HERE.parent.parent / "resources"
CONFIG_YAML = RESOURCE_DIR / "parse_data" / "config" / "perfect_simple_parse.yaml"


# Test
def test_recursive_markdown_parsing(tmp_path: pathlib.Path) -> None:
    """Ensure `Parser` captures nested Markdown files and preserves paths."""

    # 1. Build nested markdown structure under `tmp_path`
    (tmp_path / "root.md").write_text("# Root file")

    deep_dir = tmp_path / "nested" / "sub" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "leaf.md").write_text("# Leaf file")

    data_glob = str(tmp_path / "**" / "*.md")

    # 2. Run the parser with `recursive=True`
    with tempfile.TemporaryDirectory() as project_dir:
        parser = Parser(data_path_glob=data_glob, project_dir=project_dir)
        parser.start_parsing(str(CONFIG_YAML), all_files=False, recursive=True)

        # 3. Verify the Parquet output exists and contains the original paths
        output_parquet = pathlib.Path(project_dir) / "parsed_result.parquet"
        assert output_parquet.exists(), "parsed_result.parquet was not created"

        df = pd.read_parquet(output_parquet)
        parsed_paths = set(df["path"].tolist())

        expected_paths = {
            str(tmp_path / "root.md"),
            str(deep_dir / "leaf.md"),
        }

        assert expected_paths.issubset(parsed_paths), (
            "Recursive parsing failed: not all nested Markdown files were "
            "found or their paths were not preserved."
        )
