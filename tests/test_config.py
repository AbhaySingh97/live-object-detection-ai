import os
import tempfile
import pytest
from app.config import AppConfig, ModelConfig, CameraConfig


def test_default_config():
    config = AppConfig()
    assert config.model.conf_threshold == 0.35
    assert config.camera.fps == 30
    assert config.tracking.enabled is True
    assert config.alerts.line_crossing_enabled is True


def test_yaml_serialization():
    config = AppConfig()
    config.model.conf_threshold = 0.50
    config.camera.width = 1920

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        config.to_yaml(tmp_path)
        loaded = AppConfig.from_yaml(tmp_path)
        assert loaded.model.conf_threshold == 0.50
        assert loaded.camera.width == 1920
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
