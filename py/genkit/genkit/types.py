# Copyright 2022 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# generated by datamodel-codegen:
#   filename:  genkit-schema.json
#   timestamp: 2024-11-15T18:16:51+00:00

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, Field


class InstrumentationLibrary(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    version: Optional[str] = None
    schemaUrl: Optional[str] = None


class SpanContext(BaseModel):
    class Config:
        extra = Extra.forbid

    traceId: str
    spanId: str
    isRemote: Optional[bool] = None
    traceFlags: float


class SameProcessAsParentSpan(BaseModel):
    class Config:
        extra = Extra.forbid

    value: bool


class State(Enum):
    success = 'success'
    error = 'error'


class SpanMetadata(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    state: Optional[State] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    isRoot: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None


class SpanStatus(BaseModel):
    class Config:
        extra = Extra.forbid

    code: float
    message: Optional[str] = None


class Annotation(BaseModel):
    class Config:
        extra = Extra.forbid

    attributes: Dict[str, Any]
    description: str


class TimeEvent(BaseModel):
    class Config:
        extra = Extra.forbid

    time: float
    annotation: Annotation


class Code(Enum):
    blocked = 'blocked'
    other = 'other'
    unknown = 'unknown'


class CandidateError(BaseModel):
    class Config:
        extra = Extra.forbid

    index: float
    code: Code
    message: Optional[str] = None


class FinishReason(Enum):
    stop = 'stop'
    length = 'length'
    blocked = 'blocked'
    other = 'other'
    unknown = 'unknown'


class DataPart(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Optional[Any] = None
    toolRequest: Optional[Any] = None
    toolResponse: Optional[Any] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class Format(Enum):
    json = 'json'
    text = 'text'
    media = 'media'


class Output(BaseModel):
    class Config:
        extra = Extra.forbid

    format: Optional[Format] = None
    schema_: Optional[Dict[str, Any]] = Field(None, alias='schema')


class Content(BaseModel):
    class Config:
        extra = Extra.forbid

    text: str
    media: Optional[Any] = None


class Media(BaseModel):
    class Config:
        extra = Extra.forbid

    contentType: Optional[str] = None
    url: str


class Content1(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Media


class ContextItem(BaseModel):
    class Config:
        extra = Extra.forbid

    content: List[Union[Content, Content1]]
    metadata: Optional[Dict[str, Any]] = None


class GenerationCommonConfig(BaseModel):
    class Config:
        extra = Extra.forbid

    version: Optional[str] = None
    temperature: Optional[float] = None
    maxOutputTokens: Optional[float] = None
    topK: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[List[str]] = None


class GenerationUsage(BaseModel):
    class Config:
        extra = Extra.forbid

    inputTokens: Optional[float] = None
    outputTokens: Optional[float] = None
    totalTokens: Optional[float] = None
    inputCharacters: Optional[float] = None
    outputCharacters: Optional[float] = None
    inputImages: Optional[float] = None
    outputImages: Optional[float] = None
    inputVideos: Optional[float] = None
    outputVideos: Optional[float] = None
    inputAudioFiles: Optional[float] = None
    outputAudioFiles: Optional[float] = None
    custom: Optional[Dict[str, float]] = None


class Role(Enum):
    system = 'system'
    user = 'user'
    model = 'model'
    tool = 'tool'


class ToolDefinition(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(
        ..., description='Valid JSON Schema representing the input of the tool.'
    )
    outputSchema: Optional[Dict[str, Any]] = Field(
        None, description='Valid JSON Schema describing the output of the tool.'
    )


class ToolRequest1(BaseModel):
    class Config:
        extra = Extra.forbid

    ref: Optional[str] = None
    name: str
    input: Optional[Any] = None


class ToolResponse1(BaseModel):
    class Config:
        extra = Extra.forbid

    ref: Optional[str] = None
    name: str
    output: Optional[Any] = None


class Start(BaseModel):
    class Config:
        extra = Extra.forbid

    input: Optional[Any] = None
    labels: Optional[Dict[str, str]] = None


class Schedule(BaseModel):
    class Config:
        extra = Extra.forbid

    input: Optional[Any] = None
    delay: Optional[float] = None


class RunScheduled(BaseModel):
    class Config:
        extra = Extra.forbid

    flowId: str


class Retry(BaseModel):
    class Config:
        extra = Extra.forbid

    flowId: str


class Resume(BaseModel):
    class Config:
        extra = Extra.forbid

    flowId: str
    payload: Optional[Any] = None


class State1(BaseModel):
    class Config:
        extra = Extra.forbid

    flowId: str


class FlowActionInput(BaseModel):
    class Config:
        extra = Extra.forbid

    start: Optional[Start] = None
    schedule: Optional[Schedule] = None
    runScheduled: Optional[RunScheduled] = None
    retry: Optional[Retry] = None
    resume: Optional[Resume] = None
    state: Optional[State1] = None
    auth: Optional[Any] = None


class FlowError(BaseModel):
    class Config:
        extra = Extra.forbid

    error: Optional[str] = None
    stacktrace: Optional[str] = None


class FlowResponse(BaseModel):
    class Config:
        extra = Extra.forbid

    response: Any = None


class FlowResult(FlowResponse, FlowError):
    pass


class FlowStateExecution(BaseModel):
    class Config:
        extra = Extra.forbid

    startTime: Optional[float] = Field(
        None, description='start time in milliseconds since the epoch'
    )
    endTime: Optional[float] = Field(
        None, description='end time in milliseconds since the epoch'
    )
    traceIds: List[str]


class Cache(BaseModel):
    class Config:
        extra = Extra.forbid

    value: Optional[Any] = None
    empty: bool = Field(True)


class BlockedOnStep(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    schema_: Optional[str] = Field(None, alias='schema')


class Operation(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str = Field(
        ...,
        description='server-assigned name, which is only unique within the same service that originally returns it.',
    )
    metadata: Optional[Any] = Field(
        None,
        description='Service-specific metadata associated with the operation. It typically contains progress information and common metadata such as create time.',
    )
    done: Optional[bool] = Field(
        False,
        description='If the value is false, it means the operation is still in progress. If true, the operation is completed, and either error or response is available.',
    )
    result: Optional[FlowResult] = None
    blockedOnStep: Optional[BlockedOnStep] = None


class Content2(BaseModel):
    class Config:
        extra = Extra.forbid

    text: str
    media: Optional[Any] = None


class Media2(BaseModel):
    class Config:
        extra = Extra.forbid

    contentType: Optional[str] = None
    url: str


class Content3(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Media2


class Items(BaseModel):
    class Config:
        extra = Extra.forbid

    content: List[Union[Content2, Content3]]
    metadata: Optional[Dict[str, Any]] = None


class OutputModel(BaseModel):
    class Config:
        extra = Extra.forbid

    format: Optional[Format] = None
    schema_: Optional[Dict[str, Any]] = Field(None, alias='schema')


class Link(BaseModel):
    class Config:
        extra = Extra.forbid

    context: Optional[SpanContext] = None
    attributes: Optional[Dict[str, Any]] = None
    droppedAttributesCount: Optional[float] = None


class TimeEvents(BaseModel):
    class Config:
        extra = Extra.forbid

    timeEvent: Optional[List[TimeEvent]] = None


class SpanData(BaseModel):
    class Config:
        extra = Extra.forbid

    spanId: str
    traceId: str
    parentSpanId: Optional[str] = None
    startTime: float
    endTime: float
    attributes: Dict[str, Any]
    displayName: str
    links: Optional[List[Link]] = None
    instrumentationLibrary: InstrumentationLibrary
    spanKind: str
    sameProcessAsParentSpan: Optional[SameProcessAsParentSpan] = None
    status: Optional[SpanStatus] = None
    timeEvents: Optional[TimeEvents] = None
    truncated: Optional[bool] = None


class TraceData(BaseModel):
    class Config:
        extra = Extra.forbid

    traceId: str
    displayName: Optional[str] = None
    startTime: Optional[float] = None
    endTime: Optional[float] = None
    spans: Dict[str, SpanData]


class MediaPart(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Media
    toolRequest: Optional[Any] = None
    toolResponse: Optional[Any] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class Supports(BaseModel):
    class Config:
        extra = Extra.forbid

    multiturn: Optional[bool] = None
    media: Optional[bool] = None
    tools: Optional[bool] = None
    systemRole: Optional[bool] = None
    output: Optional[List[Format]] = None
    context: Optional[bool] = None


class ModelInfo(BaseModel):
    class Config:
        extra = Extra.forbid

    versions: Optional[List[str]] = None
    label: Optional[str] = None
    supports: Optional[Supports] = None


class TextPart(BaseModel):
    class Config:
        extra = Extra.forbid

    text: str
    media: Optional[Any] = None
    toolRequest: Optional[Any] = None
    toolResponse: Optional[Any] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolRequestPart(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Optional[Any] = None
    toolRequest: ToolRequest1
    toolResponse: Optional[Any] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolResponsePart(BaseModel):
    class Config:
        extra = Extra.forbid

    text: Optional[Any] = None
    media: Optional[Any] = None
    toolRequest: Optional[Any] = None
    toolResponse: ToolResponse1
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class FlowInvokeEnvelopeMessage(BaseModel):
    class Config:
        extra = Extra.forbid

    start: Optional[Start] = None
    schedule: Optional[Schedule] = None
    runScheduled: Optional[RunScheduled] = None
    retry: Optional[Retry] = None
    resume: Optional[Resume] = None
    state: Optional[State1] = None


class FlowState(BaseModel):
    class Config:
        extra = Extra.forbid

    name: Optional[str] = None
    flowId: str
    input: Optional[Any] = None
    startTime: float = Field(
        ..., description='start time in milliseconds since the epoch'
    )
    cache: Dict[str, Cache]
    eventsTriggered: Dict[str, Any]
    blockedOnStep: Optional[BlockedOnStep]
    operation: Operation
    traceContext: Optional[str] = None
    executions: List[FlowStateExecution]


class DocumentData(BaseModel):
    class Config:
        extra = Extra.forbid

    content: List[
        Union[TextPart, MediaPart, ToolRequestPart, ToolResponsePart, DataPart]
    ]
    metadata: Optional[Dict[str, Any]] = None


class GenerateResponseChunk(BaseModel):
    class Config:
        extra = Extra.forbid

    content: List[
        Union[TextPart, MediaPart, ToolRequestPart, ToolResponsePart, DataPart]
    ]
    custom: Optional[Any] = None
    aggregated: Optional[bool] = None
    index: float


class Message(BaseModel):
    class Config:
        extra = Extra.forbid

    role: Role
    content: List[
        Union[TextPart, MediaPart, ToolRequestPart, ToolResponsePart, DataPart]
    ]
    metadata: Optional[Dict[str, Any]] = None


class ModelResponseChunk(BaseModel):
    class Config:
        extra = Extra.forbid

    content: List[
        Union[TextPart, MediaPart, ToolRequestPart, ToolResponsePart, DataPart]
    ]
    custom: Optional[Any] = None
    aggregated: Optional[bool] = None


class Candidate(BaseModel):
    class Config:
        extra = Extra.forbid

    index: float
    message: Message
    usage: Optional[GenerationUsage] = None
    finishReason: FinishReason
    finishMessage: Optional[str] = None
    custom: Optional[Any] = None


class GenerateRequest(BaseModel):
    class Config:
        extra = Extra.forbid

    messages: List[Message]
    config: Optional[Any] = None
    tools: Optional[List[ToolDefinition]] = None
    output: Optional[Output] = None
    context: Optional[List[ContextItem]] = None
    candidates: Optional[float] = None


class GenerateResponse(BaseModel):
    class Config:
        extra = Extra.forbid

    message: Optional[Message] = None
    finishReason: Optional[FinishReason] = None
    finishMessage: Optional[str] = None
    latencyMs: Optional[float] = None
    usage: Optional[GenerationUsage] = None
    custom: Optional[Any] = None
    request: Optional[GenerateRequest] = None
    candidates: Optional[List[Candidate]] = None


class ModelRequest(BaseModel):
    class Config:
        extra = Extra.forbid

    messages: List[Message]
    config: Optional[Any] = None
    tools: Optional[List[ToolDefinition]] = None
    output: Optional[OutputModel] = None
    context: Optional[List[Items]] = None


class ModelResponse(BaseModel):
    class Config:
        extra = Extra.forbid

    message: Optional[Message] = None
    finishReason: FinishReason
    finishMessage: Optional[str] = None
    latencyMs: Optional[float] = None
    usage: Optional[GenerationUsage] = None
    custom: Optional[Any] = None
    request: Optional[GenerateRequest] = None
