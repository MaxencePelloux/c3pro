import pytest

from conftest import slide_name
from utils.graph_io import resolve_graph_path
from utils.pathid import extract_pathid


def test_extract_pathid_separators():
    pid = slide_name(17, 1)[:10]
    assert extract_pathid(slide_name(17, 1)) == pid
    assert extract_pathid(slide_name(17, 1).replace("-", " ", 1)) == pid
    assert extract_pathid(f"  {slide_name(17, 1)}  ") == pid


@pytest.mark.parametrize("bad", ["", "garbage", "1ZZ000001-A01", "17Z1000001-A01"])
def test_extract_pathid_rejects(bad):
    with pytest.raises(ValueError):
        extract_pathid(bad)


def test_resolve_graph_path(tmp_path):
    dash = slide_name(17, 1)
    spaced = dash.replace("-", " ", 1)
    (tmp_path / f"{dash}.pt").write_bytes(b"x")
    (tmp_path / f"{spaced.replace(' ', '')}.pt").write_bytes(b"x")
    assert resolve_graph_path(dash, tmp_path).name == f"{dash}.pt"
    assert resolve_graph_path(spaced, tmp_path).name == f"{spaced.replace(' ', '')}.pt"
    with pytest.raises(FileNotFoundError):
        resolve_graph_path(slide_name(17, 2), tmp_path)
