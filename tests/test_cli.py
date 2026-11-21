import pytest
from typer.testing import CliRunner

from sylvigie import __version__
from sylvigie.cli import app

runner = CliRunner()

PIPELINE_COMMANDS = ["detect", "evaluate"]


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "version",
        "suggest-zone",
        "grid",
        "ingest",
        "plot-series",
        *PIPELINE_COMMANDS,
    ]:
        assert command in result.output


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


@pytest.mark.parametrize("command", PIPELINE_COMMANDS)
def test_unimplemented_commands_fail_loudly(command):
    result = runner.invoke(app, [command])

    assert result.exit_code == 1
    assert "Not implemented yet" in result.output
