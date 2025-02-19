# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Dict, List, Optional, Union

import pydantic


class GoogleAiApiTypes(enum.StrEnum):
    CHAT = 'chat'


class GoogleAiApiParams(pydantic.BaseModel):
    pass


class Category(enum.StrEnum):
    ('HARM_CATEGORY_UNSPECIFIED',)
    ('HARM_CATEGORY_HATE_SPEECH',)
    ('HARM_CATEGORY_SEXUALLY_EXPLICIT',)
    ('HARM_CATEGORY_HARASSMENT',)
    ('HARM_CATEGORY_DANGEROUS_CONTENT',)


class Threshold(enum.StrEnum):
    ('BLOCK_LOW_AND_ABOVE',)
    ('BLOCK_MEDIUM_AND_ABOVE',)
    ('BLOCK_ONLY_HIGH',)
    ('BLOCK_NONE',)


class SafetySettingsSchema(pydantic.BaseModel):
    category: Category
    threshold: Threshold


class Mode(enum.StrEnum):
    ('MODE_UNSPECIFIED',)
    ('AUTO',)
    ('ANY',)
    'NONE'


class FunctionalCallingConfig(pydantic.BaseModel):
    mode: Optional[Mode]
    allowed_function_names: Optional[List]


class GoogleAiConfigSchema(pydantic.BaseModel):
    safety_settings: Optional[List[SafetySettingsSchema]]
    code_execution: Union[bool, Dict]
    context_cache: Optional[bool]
    functional_calling_config: Optional[FunctionalCallingConfig]


class GoogleAiPluginOptions:
    apiKey: Optional[str] = None
    apiVersion: Optional[Union[List[str], str]] = None
    baseUrl: Optional[str]
    # TODO models: (
    #   | ModelReference</** @ignore */ typeof GeminiConfigSchema>
    #   | string
    # )
