import tomllib
from pathlib import Path

import pytest


@pytest.mark.portable
def test_published_metadata_has_no_source_only_urls():
    root = Path(__file__).parents[2]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    project = metadata["project"]
    published_requirements = list(project["dependencies"])
    for requirements in project["optional-dependencies"].values():
        published_requirements.extend(requirements)

    assert not any("git+" in requirement for requirement in published_requirements)
    assert metadata["dependency-groups"]["facility-source"] == [
        "eiger-io @ git+https://github.com/NSLS-II-CHX/eiger-io.git@cb13bdc336e445697e6483556116aaba0368a5d3",
        "ModestImage @ git+https://github.com/ChrisBeaumont/mpl-modest-image.git@"
        "4174514a9ce7f4160fb6cbd200df6897694e0ac3",
    ]


@pytest.mark.portable
def test_obsolete_packaging_files_are_removed():
    root = Path(__file__).parents[2]
    for filename in ("setup.py", "setup.cfg", "requirements.txt", "requirements-dev.txt", "versioneer.py"):
        assert not (root / filename).exists()
