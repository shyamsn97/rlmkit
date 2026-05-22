from pathlib import Path

from rlmflow.tools.filesystem import ls


def test_ls_returns_paths_usable_by_file_tools(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    assert ls("nested") == [str((Path("nested") / "item.txt").resolve())]
    assert ls("nested/item.txt") == [str(Path("nested/item.txt").resolve())]
