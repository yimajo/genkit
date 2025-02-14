# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Exposes an API for inspecting and interacting with Genkit in development."""

import json
from typing import Any

import structlog
from genkit.core.registry import Registry
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = structlog.get_logger()


def make_reflection_server(registry: Registry) -> Starlette:
    """Returns a Starlette application for reflection."""

    async def health_check(_: Request) -> Response:
        """Health check endpoint."""
        return Response(status_code=200)

    async def list_actions(_: Request) -> JSONResponse:
        """List all available actions."""
        actions = {}
        for action_type in registry.actions:
            for name in registry.actions[action_type]:
                action = registry.lookup_action(action_type, name)
                key = f'/{action_type}/{name}'
                actions[key] = {
                    'key': key,
                    'name': action.name,
                    'inputSchema': action.input_schema,
                    'outputSchema': action.output_schema,
                    'metadata': action.metadata,
                }
        return JSONResponse(actions)

    async def notify(_: Request) -> Response:
        """Handle notifications."""
        return Response(status_code=200)

    async def run_action(request: Request) -> JSONResponse:
        """Run an action with the given input."""
        payload: dict[str, Any] = await request.json()
        logger.debug('run_action.payload', payload=payload)

        action = registry.lookup_by_absolute_name(payload['key'])
        logger.debug('run_action.action', action=action)

        if '/flow/' in payload['key']:
            input_action = action.input_type.validate_python(
                payload['input']['start']['input']
            )
        else:
            input_action = action.input_type.validate_python(payload['input'])

        output = action.fn(input_action)

        headers = {
            'x-genkit-version': '0.9.1',
        }

        if isinstance(output.response, BaseModel):
            return JSONResponse(
                {
                    'result': json.loads(output.response.model_dump_json()),
                    'traceId': output.traceId,
                },
                headers=headers,
            )

        return JSONResponse(
            {
                'result': output.response,
                'telemetry': {'traceId': output.traceId},
            },
            headers=headers,
        )

    routes = [
        Route('/api/__health', health_check),
        Route('/api/actions', list_actions),
        Route('/api/notify', notify, methods=['POST']),
        Route('/api/runAction', run_action, methods=['POST']),
    ]

    return Starlette(routes=routes)
