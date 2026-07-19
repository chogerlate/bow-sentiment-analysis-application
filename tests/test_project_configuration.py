"""Tests for installable project entry points."""

from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path

from omegaconf import OmegaConf


def test_training_console_script_resolves_existing_module() -> None:
    """Point the installed training command at the real main module."""
    (training_script,) = entry_points(
        group="console_scripts",
        name="sentiment-train",
    )

    assert training_script.value == "sentiment_analysis.main:main"
    assert callable(training_script.load())
    packaged_config = files("sentiment_analysis.configs").joinpath("config.yaml")
    project_config = Path(__file__).parents[1] / "configs" / "config.yaml"

    assert packaged_config.is_file()
    assert packaged_config.read_bytes() == project_config.read_bytes()


def test_training_paths_are_portable() -> None:
    """Keep project defaults independent from a contributor's home directory."""
    project_config = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = OmegaConf.load(project_config)

    assert config.paths.artifacts == "artifacts"
    assert config.paths.models == "artifacts/models"
    assert config.paths.dataset == "dataset"
    assert config.paths.train_data == "dataset/train.csv"
    assert config.paths.test_data == "dataset/test.csv"
