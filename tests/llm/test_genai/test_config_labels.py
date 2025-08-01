"""Tests for Google GenAI config labels preservation."""

import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from instructor.providers.gemini.utils import (
    handle_genai_structured_outputs,
    handle_genai_tools,
)


class SimpleModel(BaseModel):
    """A simple test model."""
    name: str
    age: int


class TestConfigLabelsPreservation:
    """Test that config labels are preserved when creating GenerateContentConfig."""

    def test_handle_genai_structured_outputs_preserves_dict_labels(self):
        """Test that handle_genai_structured_outputs preserves labels from dict config."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {
                    "environment": "development",
                    "tenant": "test_tenant"
                }
            }
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that labels are preserved in the config
        assert hasattr(config, "labels")
        assert config.labels == {
            "environment": "development",
            "tenant": "test_tenant"
        }

    def test_handle_genai_tools_preserves_dict_labels(self):
        """Test that handle_genai_tools preserves labels from dict config."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {
                    "environment": "production",
                    "version": "1.0.0"
                }
            }
        }
        
        response_model, result_kwargs = handle_genai_tools(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that labels are preserved in the config
        assert hasattr(config, "labels")
        assert config.labels == {
            "environment": "production",
            "version": "1.0.0"
        }

    def test_handle_genai_structured_outputs_preserves_object_labels(self):
        """Test that handle_genai_structured_outputs preserves labels from object config."""
        # Mock GenerateContentConfig-like object with labels
        mock_config = MagicMock()
        mock_config.labels = {
            "project": "instructor-test",
            "stage": "testing"
        }
        
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": mock_config
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that labels are preserved in the config
        assert hasattr(config, "labels")
        assert config.labels == {
            "project": "instructor-test",
            "stage": "testing"
        }

    def test_handle_genai_tools_preserves_object_labels(self):
        """Test that handle_genai_tools preserves labels from object config."""
        # Mock GenerateContentConfig-like object with labels
        mock_config = MagicMock()
        mock_config.labels = {
            "team": "ai-research",
            "cost_center": "123456"
        }
        
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": mock_config
        }
        
        response_model, result_kwargs = handle_genai_tools(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that labels are preserved in the config
        assert hasattr(config, "labels")
        assert config.labels == {
            "team": "ai-research",
            "cost_center": "123456"
        }

    def test_handle_genai_structured_outputs_no_labels_config(self):
        """Test that handle_genai_structured_outputs works when no config is provided."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}]
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists (created by the function)
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that no labels are set (or labels is None/empty)
        if hasattr(config, "labels"):
            assert config.labels is None or config.labels == {}

    def test_handle_genai_tools_no_labels_config(self):
        """Test that handle_genai_tools works when no config is provided."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}]
        }
        
        response_model, result_kwargs = handle_genai_tools(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists (created by the function)
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that no labels are set (or labels is None/empty)
        if hasattr(config, "labels"):
            assert config.labels is None or config.labels == {}

    def test_handle_genai_structured_outputs_config_without_labels(self):
        """Test that handle_genai_structured_outputs works when config exists but has no labels."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "other_param": "some_value"
            }
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists (created by the function)
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that no labels are set (or labels is None/empty)
        if hasattr(config, "labels"):
            assert config.labels is None or config.labels == {}

    def test_handle_genai_tools_config_without_labels(self):
        """Test that handle_genai_tools works when config exists but has no labels."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "other_param": "some_value"
            }
        }
        
        response_model, result_kwargs = handle_genai_tools(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists (created by the function)
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that no labels are set (or labels is None/empty)
        if hasattr(config, "labels"):
            assert config.labels is None or config.labels == {}

    def test_handle_genai_structured_outputs_empty_labels(self):
        """Test that handle_genai_structured_outputs handles empty labels dict."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {}
            }
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that empty labels are preserved
        assert hasattr(config, "labels")
        assert config.labels == {}

    def test_handle_genai_tools_empty_labels(self):
        """Test that handle_genai_tools handles empty labels dict."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {}
            }
        }
        
        response_model, result_kwargs = handle_genai_tools(
            SimpleModel, new_kwargs
        )
        
        # Check that the response model is returned correctly
        assert response_model == SimpleModel
        
        # Check that config exists and contains the GenerateContentConfig
        assert "config" in result_kwargs
        config = result_kwargs["config"]
        
        # Check that empty labels are preserved
        assert hasattr(config, "labels")
        assert config.labels == {}

    def test_handle_genai_structured_outputs_none_response_model(self):
        """Test that handle_genai_structured_outputs works with None response_model."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {
                    "test": "value"
                }
            }
        }
        
        response_model, result_kwargs = handle_genai_structured_outputs(
            None, new_kwargs
        )
        
        # When response_model is None, it should just do message conversion
        assert response_model is None
        # The original config should be preserved in some form (check contents exist)
        assert "contents" in result_kwargs

    def test_handle_genai_tools_none_response_model(self):
        """Test that handle_genai_tools works with None response_model."""
        new_kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "config": {
                "labels": {
                    "test": "value"
                }
            }
        }
        
        response_model, result_kwargs = handle_genai_tools(
            None, new_kwargs
        )
        
        # When response_model is None, it should just do message conversion
        assert response_model is None
        # The original config should be preserved in some form (check contents exist)
        assert "contents" in result_kwargs