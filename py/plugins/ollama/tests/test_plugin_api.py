# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import ollama as ollama_api

from genkit.core.types import Message, Role, TextPart, GenerateResponse
from genkit.veneer import Genkit


def test_adding_ollama_chat_model_to_genkit_veneer(
    ollama_model: str,
    genkit_veneer_chat_model: Genkit,
):
    assert len(genkit_veneer_chat_model.registry.actions) == 1
    assert ollama_model in genkit_veneer_chat_model.registry.actions['model']


def test_adding_ollama_generation_model_to_genkit_veneer(
    ollama_model: str,
    genkit_veneer_generate_model: Genkit,
):
    assert len(genkit_veneer_generate_model.registry.actions) == 1
    assert (
        ollama_model in genkit_veneer_generate_model.registry.actions['model']
    )


def test_get_chat_model_response_from_llama_api_flow(
    mock_ollama_api_client: mock.Mock, genkit_veneer_chat_model: Genkit
):
    mock_response_message = 'Mocked response message'

    mock_ollama_api_client.return_value.chat.return_value = (
        ollama_api.ChatResponse(
            message=ollama_api.Message(
                content=mock_response_message,
                role='user',
            )
        )
    )

    def _test_fun():
        return genkit_veneer_chat_model.generate(
            messages=[
                Message(
                    role=Role.user,
                    content=[
                        TextPart(text='Test message'),
                    ],
                )
            ]
        )

    response = genkit_veneer_chat_model.flow()(_test_fun)()

    assert isinstance(response, GenerateResponse)
    assert response.message.content[0].text == mock_response_message


def test_get_generate_model_response_from_llama_api_flow(
    mock_ollama_api_client: mock.Mock, genkit_veneer_generate_model: Genkit
):
    mock_response_message = 'Mocked response message'

    mock_ollama_api_client.return_value.generate.return_value = (
        ollama_api.GenerateResponse(
            response=mock_response_message,
        )
    )

    def _test_fun():
        return genkit_veneer_generate_model.generate(
            messages=[
                Message(
                    role=Role.user,
                    content=[
                        TextPart(text='Test message'),
                    ],
                )
            ]
        )

    response = genkit_veneer_generate_model.flow()(_test_fun)()

    assert isinstance(response, GenerateResponse)
    assert response.message.content[0].text == mock_response_message
