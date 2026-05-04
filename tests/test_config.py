from pathlib import Path

from src.config import load_config


def test_load_config_keeps_llm_and_embedding_settings_separate(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "llm_base_url: https://llm.example/v1",
                "llm_api_key: llm-key",
                "llm_model_name: llm-model",
                "embedding_base_url: https://embed.example/v1",
                "embedding_api_key: embed-key",
                "embedding_model_name: embed-model",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.llm_base_url == "https://llm.example/v1"
    assert config.llm_api_key == "llm-key"
    assert config.llm_model_name == "llm-model"
    assert config.embedding_base_url == "https://embed.example/v1"
    assert config.embedding_api_key == "embed-key"
    assert config.embedding_model_name == "embed-model"


def test_config_does_not_expose_old_openai_aliases():
    config = load_config()

    assert not hasattr(config, "openai_base_url")
    assert not hasattr(config, "openai_api_key")
    assert not hasattr(config, "model_name")
    assert not hasattr(config, "embedding_model")
