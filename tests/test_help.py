import subprocess
import sys

import pytest

from conftest import SCRIPTS

SCRIPT_FILES = sorted(SCRIPTS.glob("[01][0-9]_*.py"))
# 04, 10, 13, 15, 16 and 17 are run for real in the other tests
HELP_ONLY = [p for p in SCRIPT_FILES if p.name[:2] not in {"04", "10", "13", "15", "16", "17"}]


def test_all_steps_present():
    assert [p.name[:2] for p in SCRIPT_FILES] == [f"{i:02d}" for i in range(1, 18)]


@pytest.mark.parametrize("path", HELP_ONLY, ids=lambda p: p.name[:2])
def test_help(path):
    r = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()
