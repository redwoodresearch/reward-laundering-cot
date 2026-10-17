"""analysis/build_figures.py renders the post's figure page from the committed template and figure data."""

import hashlib
import shutil
import subprocess

import pytest

from analysis.build_figures import HERE, build

# sha256 of the figure page as last published to the post's figures artifact (version 42, 2026-10-17: these exact bytes,
# inside the host's HTML skeleton); it changes only with an intended change to a figure or caption, then republished
PUBLISHED = "ad1caedc0d0eccaf614cdc81848fda96ef5a87252265598bcb49f7dbe68b9561"


def test_the_intermediates_rebuild_every_figure_data_file_exactly():
    from analysis.compute import figures

    assert figures.build() == {fig: (figures.FIGURES / f"{fig}.json").read_text() for fig in figures.BUILDERS}


def test_the_committed_data_rebuild_the_published_page_exactly():
    assert hashlib.sha256(build(HERE / "figures" / "data").encode()).hexdigest() == PUBLISHED


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_every_figure_draws_without_errors(tmp_path):
    page = tmp_path / "figures.html"
    page.write_text(build(HERE / "figures" / "data"))
    result = subprocess.run(["node", str(HERE / "check_figures.js"), str(page)], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("chars of SVG") == 8


def test_missing_figure_data_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no data"):
        build(tmp_path)
