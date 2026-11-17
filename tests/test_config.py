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


def test_bbox_must_be_ordered(tmp_path, raw_config):
    raw_config["bbox"] = [10, 0, 0, 10]

    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, raw_config))


def test_window_must_be_whole_number_of_steps(tmp_path, raw_config):
    raw_config["selection"]["window_km"] = 25
    raw_config["selection"]["step_km"] = 2

    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, raw_config))
