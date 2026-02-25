"""Tests for notebook whitelist configuration parsing."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from joplin_mcp.config import JoplinMCPConfig


class TestNotebookWhitelistConfig:
    """Test notebook whitelist configuration loading from various sources."""

    def test_whitelist_parses_from_json_config(self, tmp_path):
        """Test that notebook_whitelist is correctly parsed from a JSON config file."""
        config_data = {
            "token": "test-token-1234567890",
            "notebook_whitelist": ["Projects/*", "Work/**", "!Work/Secret"],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = JoplinMCPConfig.from_file(config_file)

        assert config.notebook_whitelist == ["Projects/*", "Work/**", "!Work/Secret"]

    def test_whitelist_parses_from_yaml_config(self, tmp_path):
        """Test that notebook_whitelist is correctly parsed from a YAML config file."""
        config_data = {
            "token": "test-token-1234567890",
            "notebook_whitelist": ["Projects/*", "Personal/Journal"],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = JoplinMCPConfig.from_file(config_file)

        assert config.notebook_whitelist == ["Projects/*", "Personal/Journal"]

    def test_whitelist_parses_from_environment(self):
        """Test that JOPLIN_NOTEBOOK_WHITELIST env var is parsed as comma-separated list."""
        with patch.dict(
            os.environ,
            {"JOPLIN_NOTEBOOK_WHITELIST": "Projects/*,Work/**,!Work/Secret"},
            clear=False,
        ):
            config = JoplinMCPConfig.from_environment()

        assert config.notebook_whitelist == ["Projects/*", "Work/**", "!Work/Secret"]

    def test_whitelist_env_strips_whitespace(self):
        """Test that whitespace is stripped from comma-separated env var entries."""
        with patch.dict(
            os.environ,
            {"JOPLIN_NOTEBOOK_WHITELIST": " Projects/* , Work/** , !Work/Secret "},
            clear=False,
        ):
            config = JoplinMCPConfig.from_environment()

        assert config.notebook_whitelist == ["Projects/*", "Work/**", "!Work/Secret"]

    def test_whitelist_defaults_to_none(self):
        """Test that notebook_whitelist defaults to None when not configured."""
        config = JoplinMCPConfig(token="test-token-1234567890")

        assert config.notebook_whitelist is None

    def test_whitelist_defaults_to_none_from_environment(self):
        """Test that notebook_whitelist is None when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = JoplinMCPConfig.from_environment()

        assert config.notebook_whitelist is None

    def test_has_notebook_whitelist_true(self):
        """Test has_notebook_whitelist returns True for non-empty whitelist."""
        config = JoplinMCPConfig(
            token="test-token-1234567890",
            notebook_whitelist=["Projects/*"],
        )

        assert config.has_notebook_whitelist is True

    def test_has_notebook_whitelist_false_none(self):
        """Test has_notebook_whitelist returns False when whitelist is None."""
        config = JoplinMCPConfig(token="test-token-1234567890")

        assert config.has_notebook_whitelist is False

    def test_has_notebook_whitelist_false_empty(self):
        """Test has_notebook_whitelist returns False for empty whitelist list."""
        config = JoplinMCPConfig(
            token="test-token-1234567890",
            notebook_whitelist=[],
        )

        assert config.has_notebook_whitelist is False

    def test_whitelist_in_to_dict(self):
        """Test that to_dict includes notebook_whitelist."""
        patterns = ["Projects/*", "Work/**"]
        config = JoplinMCPConfig(
            token="test-token-1234567890",
            notebook_whitelist=patterns,
        )

        result = config.to_dict()

        assert "notebook_whitelist" in result
        assert result["notebook_whitelist"] == patterns

    def test_whitelist_none_in_to_dict(self):
        """Test that to_dict includes notebook_whitelist as None when not configured."""
        config = JoplinMCPConfig(token="test-token-1234567890")

        result = config.to_dict()

        assert "notebook_whitelist" in result
        assert result["notebook_whitelist"] is None
