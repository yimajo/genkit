# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from unittest import mock

from genkit.plugins.ollama import Ollama
from genkit.plugins.ollama.models import (
    OllamaPluginParams,
    ModelDefinition,
    OllamaAPITypes,
)
from genkit.plugins.ollama.plugin_api import ollama_api
from genkit.veneer import Genkit


@pytest.fixture
def ollama_model() -> str:
    return 'ollama/llama3.2:latest'


@pytest.fixture
def chat_model_plugin_params(ollama_model: str) -> OllamaPluginParams:
    return OllamaPluginParams(
        models=[
            ModelDefinition(
                name=ollama_model.split('/')[-1],
                api_type=OllamaAPITypes.CHAT,
            )
        ],
    )


@pytest.fixture
def genkit_veneer_chat_model(
    ollama_model: str,
    chat_model_plugin_params: OllamaPluginParams,
) -> Genkit:
    return Genkit(
        plugins=[
            Ollama(
                plugin_params=chat_model_plugin_params,
            )
        ],
        model=ollama_model,
    )


@pytest.fixture
def generate_model_plugin_params(ollama_model: str) -> OllamaPluginParams:
    return OllamaPluginParams(
        models=[
            ModelDefinition(
                name=ollama_model.split('/')[-1],
                api_type=OllamaAPITypes.GENERATE,
            )
        ],
    )


@pytest.fixture
def genkit_veneer_generate_model(
    ollama_model: str,
    generate_model_plugin_params: OllamaPluginParams,
) -> Genkit:
    return Genkit(
        plugins=[
            Ollama(
                plugin_params=generate_model_plugin_params,
            )
        ],
        model=ollama_model,
    )


@pytest.fixture
def mock_ollama_api_client():
    with mock.patch.object(ollama_api, 'Client') as mock_ollama_client:
        yield mock_ollama_client
