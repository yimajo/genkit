# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

from genkit.plugins.ollama.plugin_api import Ollama


def package_name() -> str:
    return 'genkit.plugins.ollama'


__all__ = [
    package_name.__name__,
    Ollama.__name__,
]
