# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Veneer user-facing API for application developers who use the SDK."""

import logging
import os
import threading
from collections.abc import Callable
from http.server import HTTPServer
from typing import Any

from genkit.ai.embedding import EmbedderFn, EmbedRequest, EmbedResponse
from genkit.ai.model import ModelFn
from genkit.ai.prompt import PromptFn
from genkit.core.action import Action, ActionKind
from genkit.core.plugin_abc import Plugin
from genkit.core.reflection import make_reflection_server
from genkit.core.registry import Registry
from genkit.core.schema_types import GenerateRequest, GenerateResponse, Message
from genkit.veneer import server

DEFAULT_REFLECTION_SERVER_SPEC = server.ServerSpec(
    scheme='http', host='127.0.0.1', port=3100
)

logger = logging.getLogger(__name__)


class Genkit:
    """Veneer user-facing API for application developers who use the SDK."""

    registry: Registry = Registry()

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        model: str | None = None,
        reflection_server_spec=DEFAULT_REFLECTION_SERVER_SPEC,
    ) -> None:
        self.model = model

        if server.is_dev_environment():
            runtimes_dir = os.path.join(os.getcwd(), '.genkit/runtimes')
            server.create_runtime(
                runtime_dir=runtimes_dir,
                reflection_server_spec=reflection_server_spec,
                at_exit_fn=os.remove,
            )
            self.thread = threading.Thread(
                target=self.start_server,
                args=(
                    reflection_server_spec.host,
                    reflection_server_spec.port,
                ),
            )
            self.thread.start()

        if not plugins:
            logger.warning('No plugins provided to Genkit')
        else:
            for plugin in plugins:
                if isinstance(plugin, Plugin):
                    plugin.attach_to_veneer(veneer=self)
                else:
                    raise ValueError(
                        f'Invalid {plugin=} provided to Genkit: '
                        f'must be of type `genkit.core.plugin_abc.Plugin`'
                    )

    def start_server(self, host: str, port: int) -> None:
        httpd = HTTPServer((host, port), make_reflection_server(self.registry))
        httpd.serve_forever()

    def generate(
        self,
        model: str | None = None,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system: str | None = None,
        tools: list[str] | None = None,
    ) -> GenerateResponse:
        model = model if model is not None else self.model
        if model is None:
            raise Exception('No model configured.')

        model_action = self.registry.lookup_action(ActionKind.MODEL, model)

        return model_action.fn(GenerateRequest(messages=messages)).response

    def embed(
        self, model: str | None = None, documents: list[str] | None = None
    ) -> EmbedResponse:
        embed_action = self.registry.lookup_action(ActionKind.EMBEDDER, model)

        return embed_action.fn(EmbedRequest(documents=documents)).response

    def flow(self, name: str | None = None) -> Callable[[Callable], Callable]:
        def wrapper(func: Callable) -> Callable:
            flow_name = name if name is not None else func.__name__
            action = Action(
                name=flow_name,
                kind=ActionKind.FLOW,
                fn=func,
                span_metadata={'genkit:metadata:flow:name': flow_name},
            )
            self.registry.register_action(action)

            def decorator(*args: Any, **kwargs: Any) -> GenerateResponse:
                return action.fn(*args, **kwargs).response

            return decorator

        return wrapper

    def define_model(
        self,
        name: str,
        fn: ModelFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        action = Action(
            name=name, kind=ActionKind.MODEL, fn=fn, metadata=metadata
        )
        self.registry.register_action(action)

    def define_prompt(
        self,
        name: str,
        fn: PromptFn,
        model: str | None = None,
    ) -> Callable[[Any | None], GenerateResponse]:
        def prompt(input_prompt: Any | None = None) -> GenerateResponse:
            req = fn(input_prompt)
            return self.generate(messages=req.messages, model=model)

        action = Action(kind=ActionKind.PROMPT, name=name, fn=fn)
        self.registry.register_action(action)

        def wrapper(input_prompt: Any | None = None) -> GenerateResponse:
            return action.fn(input_prompt)

        return wrapper

    def define_embedder(
        self,
        name: str,
        fn: EmbedderFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        action = Action(
            name=name, kind=ActionKind.EMBEDDER, fn=fn, metadata=metadata
        )
        self.registry.register_action(action)
