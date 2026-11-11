import pytest

from synthetic import SyntheticOptions, make_capture


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch):
    """No test reaches the real `claude`; tests of the vision stage point ROOM_MODEL_CLAUDE at tests/fake_claude.py."""
    monkeypatch.setenv("ROOM_MODEL_CLAUDE", "/nonexistent/claude")


@pytest.fixture
def build(tmp_path):
    """Factory: build(**options) -> (capture_dir, truth); each call gets its own folder."""
    count = 0

    def _build(**options):
        nonlocal count
        count += 1
        root = tmp_path / f"capture-{count}"
        return root, make_capture(root, SyntheticOptions(**options))

    return _build


@pytest.fixture
def capture(build):
    return build()
