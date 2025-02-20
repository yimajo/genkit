# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The registry is used to store and lookup resources such as actions and
flows."""

from typing import Any

from genkit.ai.model import ModelFn
from genkit.core.action import Action, ActionKind


class Registry:
    """Stores actions, trace stores, flow state stores, plugins, and schemas."""

    actions: dict[ActionKind, dict[str, Action]] = {}

    @classmethod
    def register_model(
        cls,
        name: str,
        fn: ModelFn,
        metadata: dict[str, Any] | None = None,
    ):
        action = Action(
            name=name, kind=ActionKind.MODEL, fn=fn, metadata=metadata
        )
        cls.register_action(action=action)

    @classmethod
    def register_action(cls, action: Action) -> None:
        """Register an action.

        Args:
            action: The action to register.
        """
        kind = action.kind
        if kind not in cls.actions:
            cls.actions[kind] = {}
        cls.actions[kind][action.name] = action

    @classmethod
    def lookup_action(cls, kind: ActionKind, name: str) -> Action | None:
        """Lookup an action by its kind and name.

        Args:
            kind: The kind of the action.
            name: The name of the action.

        Returns:
            The action if found, otherwise None.
        """
        if kind in cls.actions and name in cls.actions[kind]:
            return cls.actions[kind][name]

    @classmethod
    def lookup_action_by_key(cls, key: str) -> Action | None:
        """Lookup an action by its key.

        The key is of the form:
        <kind>/<name>

        Args:
            key: The key to lookup the action by.

        Returns:
            The action if found, otherwise None.

        Raises:
            ValueError: If the key format is invalid.
        """
        # TODO: Use pattern matching to validate the key format
        # and verify whether the key can have only 2 parts.
        tokens = key.split('/')
        if len(tokens) != 2:
            msg = (
                f'Invalid action key format: `{key}`. '
                'Expected format: `<kind>/<name>`'
            )
            raise ValueError(msg)
        kind, name = tokens
        return cls.lookup_action(kind, name)
