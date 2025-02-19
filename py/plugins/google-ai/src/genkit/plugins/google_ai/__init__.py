# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0


"""
Google AI Plugin for Genkit
"""

from genkit.plugins.google_ai.google_ai import GoogleAi
from genkit.plugins.google_ai.schemas import GoogleAiPluginOptions


def package_name() -> str:
    return 'genkit.plugins.google_ai'


__all__ = ['package_name', 'GoogleAi', 'GoogleAiPluginOptions']
