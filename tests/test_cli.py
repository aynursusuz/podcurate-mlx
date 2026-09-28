import pytest

from podcurate_mlx import cli


def test_selection_errors_exit_nonzero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["podcurate-mlx", "select", "run.sqlite",
                                    "--policy", "policy.json", "--out", "selection"])
    monkeypatch.setattr(cli, "select", lambda *args: {"accept": 0, "error": 1})
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
