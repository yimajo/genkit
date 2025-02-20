# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from genkit.ai.embedding import EmbedRequest, EmbedResponse
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class EmbeddingModels(StrEnum):
    GECKO_001_ENG = 'textembedding-gecko@001'
    GECKO_003_ENG = 'textembedding-gecko@003'
    TEXT_EMBEDDING_004_ENG = 'text-embedding-004'
    TEXT_EMBEDDING_005_ENG = 'text-embedding-005'
    GECKO_MULTILINGUAL = 'textembedding-gecko-multilingual@001'
    TEXT_EMBEDDING_002_MULTILINGUAL = 'text-multilingual-embedding-002'


class TaskType(StrEnum):
    SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY'
    CLASSIFICATION = 'CLASSIFICATION'
    CLUSTERING = 'CLUSTERING'
    RETRIEVAL_DOCUMENT = 'RETRIEVAL_DOCUMENT'
    RETRIEVAL_QUERY = 'RETRIEVAL_QUERY'
    QUESTION_ANSWERING = 'QUESTION_ANSWERING'
    FACT_VERIFICATION = 'FACT_VERIFICATION'
    CODE_RETRIEVAL_QUERY = 'CODE_RETRIEVAL_QUERY'


class Embedder:
    DIMENSIONALITY = 128
    TASK: str = TaskType.RETRIEVAL_QUERY.value

    def __init__(self, version):
        self._version = version

    @property
    def embedding_model(self) -> TextEmbeddingModel:
        return TextEmbeddingModel.from_pretrained(self._version)

    def handle_request(self, request: EmbedRequest) -> EmbedResponse:
        inputs = [
            TextEmbeddingInput(text, self.TASK) for text in request.documents
        ]
        vertexai_embeddings = self.embedding_model.get_embeddings(inputs)
        embeddings = [embedding.values for embedding in vertexai_embeddings]
        return EmbedResponse(embeddings=embeddings)
