from rlmflow.tools.filesystem import ls


def test_ls_returns_workspace_relative_paths_for_relative_inputs(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    assert ls(".") == ["nested"]
    assert ls("nested") == ["nested/item.txt"]
    assert ls("nested/item.txt") == ["nested/item.txt"]


def test_ls_preserves_absolute_paths_for_absolute_inputs(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "item.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    assert ls(str(tmp_path / "nested")) == [str(tmp_path / "nested" / "item.txt")]
    assert ls(str(tmp_path / "nested" / "item.txt")) == [str(tmp_path / "nested" / "item.txt")]
