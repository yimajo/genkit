# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
import pytest

from genkit.core.action import ActionKind, Action, ActionExecutionContext
from genkit.core.schema_types import GenerateRequest

from . import conftest


def test_action_enum_behaves_like_str() -> None:
    """Ensure the ActionType behaves like a string and to ensure we're using the
    correct variants."""
    assert ActionKind.CHATLLM == 'chat-llm'
    assert ActionKind.CUSTOM == 'custom'
    assert ActionKind.EMBEDDER == 'embedder'
    assert ActionKind.EVALUATOR == 'evaluator'
    assert ActionKind.FLOW == 'flow'
    assert ActionKind.INDEXER == 'indexer'
    assert ActionKind.MODEL == 'model'
    assert ActionKind.PROMPT == 'prompt'
    assert ActionKind.RETRIEVER == 'retriever'
    assert ActionKind.TEXTLLM == 'text-llm'
    assert ActionKind.TOOL == 'tool'
    assert ActionKind.UTIL == 'util'


@pytest.mark.parametrize(
    "test_context",
    [
        (None, ),
        (ActionExecutionContext(),),
    ]
)
def test_action_model_with_and_without_fn_context(mock_generate_request, test_context) -> None:
    def _mock_action_callback(request: GenerateRequest,
                              context: ActionExecutionContext) -> None:
        assert request == mock_generate_request
        assert context == test_context
        conftest.action_callback_event.set()

    action = Action(
        kind=ActionKind.MODEL,
        name='test_model',
        fn=_mock_action_callback,
    )
    action.fn(request=mock_generate_request, context=test_context)
    assert conftest.action_callback_event.is_set()
