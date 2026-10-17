from pathlib import Path

import pytest

CONFIGS = sorted((Path(__file__).parent / "configs").glob("*.yaml"))


@pytest.fixture(params=CONFIGS, ids=lambda p: p.stem)
def config_path(request):
    return request.param


@pytest.fixture
def configs():
    return {p.stem: p for p in CONFIGS}
