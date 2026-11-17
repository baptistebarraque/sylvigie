from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sylvigie.config import load_config

REPO_CONFIGS = Path(__file__).parents[1] / "configs"


def _write(tmp_path, data):
    path = tmp_path / "zone.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def raw_config():
    with open(REPO_CONFIGS / "morvan.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("path", sorted(REPO_CONFIGS.glob("*.yaml")), ids=str)
def test_repo_configs_are_valid(path):
    load_config(path)


def test_unknown_key_is_rejected(tmp_path, raw_config):
    raw_config["selection"]["min_cut_per_year"] = 10  # typo

    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, raw_config))


def test_windows_must_be_ordered(tmp_path, raw_config):
    raw_config["windows"]["dev"] = [10, 0, 0, 10]

    with pytest.raises(ValidationError, match="xmin, ymin, xmax, ymax"):
        load_config(_write(tmp_path, raw_config))


def test_windows_must_not_overlap(tmp_path, raw_config):
    raw_config["windows"]["dev"] = [0, 0, 1000, 1000]
    raw_config["windows"]["eval"] = [500, 500, 1500, 1500]

    with pytest.raises(ValidationError, match="must not overlap"):
        load_config(_write(tmp_path, raw_config))


def test_touching_windows_are_allowed(tmp_path, raw_config):
    raw_config["windows"]["dev"] = [0, 0, 1000, 1000]
    raw_config["windows"]["eval"] = [1000, 0, 2000, 1000]

    assert load_config(_write(tmp_path, raw_config)).windows.eval[0] == 1000


def test_windows_must_sit_on_pixel_grid(tmp_path, raw_config):
    raw_config["windows"]["dev"] = [5, 0, 1005, 1000]

    with pytest.raises(ValidationError, match="multiples of resolution"):
        load_config(_write(tmp_path, raw_config))


def test_windows_are_optional(tmp_path, raw_config):
    del raw_config["windows"]

    assert load_config(_write(tmp_path, raw_config)).windows is None


def test_window_must_be_whole_number_of_steps(tmp_path, raw_config):
    raw_config["selection"]["window_km"] = 25
    raw_config["selection"]["step_km"] = 2

    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, raw_config))
