# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for Genkit plugins."""

from __future__ import annotations

import abc

from genkit.core.registry import Registry
from genkit.core.schema_types import GenerateRequest, GenerateResponse


class Plugin(abc.ABC):
    """
    Abstract class defining common interface
    for the Genkit Plugin implementation

    NOTE: Any plugin defined for the Genkit must inherit from this class
    """

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Entrypoint for initializing the plugin instance in Genkit

        Returns:
            None
        """
        pass

    def _register_model(
        self,
        name: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Defines plugin's model in the Genkit Registry

        Uses self._model_callback as a generic callback wrapper

        Args:
            name: name of the model to attach
            metadata: metadata information associated
                      with the provided model (optional)

        Returns:
            None
        """
        if not metadata:
            metadata = {}
        Registry.register_model(
            name=name, fn=self._model_callback, metadata=metadata
        )

    @abc.abstractmethod
    def _model_callback(self, request: GenerateRequest) -> GenerateResponse:
        """
        Wrapper around any plugin's model callback.

        Is considered an entrypoint for any model's request.

        Args:
            request: incoming request as generic
                     `genkit.core.schemas.GenerateRequest` instance

        Returns:
            Model response represented as generic
            `genkit.core.schemas.GenerateResponse` instance
        """
        pass
