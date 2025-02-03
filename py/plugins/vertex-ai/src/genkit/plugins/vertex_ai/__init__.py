# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0


"""
Google Cloud Vertex AI Plugin for Genkit.
"""

from collections.abc import Callable
from typing import Optional

import vertexai
from genkit.core.types import (
    GenerateRequest,
    GenerateResponse,
    Message,
    TextPart,
)
from genkit.veneer import Genkit
from vertexai.generative_models import Content, GenerativeModel, Part


def package_name() -> str:
    return 'genkit.plugins.vertex_ai'


def vertexAI(project_id: str | None = None) -> Callable[[Genkit], None]:
    def plugin(ai: Genkit) -> None:
        vertexai.init(location='us-central1', project=project_id)

        def gemini(request: GenerateRequest) -> GenerateResponse:
            geminiMsgs: list[Content] = []
            for m in request.messages:
                geminiParts: list[Part] = []
                for p in m.content:
                    if p.text is not None:
                        geminiParts.append(Part.from_text(p.text))
                    else:
                        raise Exception('unsupported part type')
                geminiMsgs.append(Content(role=m.role.value, parts=geminiParts))
            model = GenerativeModel('gemini-1.5-flash-002')
            response = model.generate_content(contents=geminiMsgs)
            return GenerateResponse(
                message=Message(
                    role='model', content=[TextPart(text=response.text)]
                )
            )

        ai.define_model(
            name='vertexai/gemini-1.5-flash',
            fn=gemini,
            metadata={
                'model': {
                    'label': 'banana',
                    'supports': {'multiturn': True},
                }
            },
        )

    return plugin


def gemini(name: str) -> str:
    return f'vertexai/{name}'


__all__ = ['gemini', 'package_name', 'vertexAI']
