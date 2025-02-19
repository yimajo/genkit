# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

import os

from genkit.core import plugin_abc
from genkit.core import schema_types as core_schemas
from genkit.core.schema_types import GenerateRequest, GenerateResponse
from genkit.plugins.google_ai import schemas
from genkit.veneer import veneer
from google import genai


class GoogleAi(plugin_abc.Plugin):
    def __init__(self, plugin_params=schemas.GoogleAiPluginOptions):
        api_key = (
            plugin_params.api_key
            if plugin_params.api_key
            else os.getenv('GEMINI_API_KEY')
        )
        if not api_key:
            raise ValueError(
                'Gemini api key should be passed in plugin params '
                'or as a GEMINI_API_KEY environment variable'
            )
        self._client = genai.Client(api_key=api_key)
        self._options = plugin_params

    def attach_to_veneer(self, ai: veneer.Genkit):
        ai.define_model(name='gemini-2.0-flash', fn=self._model_callback)

    def _model_callback(self, request: GenerateRequest) -> GenerateResponse:
        reqest_msgs: list[genai.types.Content] = []
        for msg in request.messages:
            message_parts: list[genai.types.Part] = []
            for p in msg.content:
                message_parts.append(
                    genai.types.Part.from_text(text=p.root.text)
                )
            reqest_msgs.append(
                genai.types.Content(parts=message_parts, role=msg.role)
            )
        response = self._client.models.generate_content(
            model='gemini-2.0-flash', contents=reqest_msgs
        )

        return GenerateResponse(
            message=core_schemas.Message(
                role=core_schemas.Role.model,
                content=[core_schemas.TextPart(text=response.text)],
            )
        )
