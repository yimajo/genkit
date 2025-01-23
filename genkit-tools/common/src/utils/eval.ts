/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { randomUUID } from 'crypto';
import { createReadStream } from 'fs';
import { readFile } from 'fs/promises';
import * as inquirer from 'inquirer';
import { createInterface } from 'readline';
import { RuntimeManager } from '../manager';
import { EvalField } from '../plugin';
import {
  Dataset,
  DatasetSchema,
  EvalInput,
  EvalInputDataset,
  EvalInputDatasetSchema,
  EvaluationDatasetSchema,
  EvaluationSample,
  EvaluationSampleSchema,
  InferenceDatasetSchema,
  InferenceSample,
  InferenceSampleSchema,
} from '../types';
import { Action } from '../types/action';
import { DocumentData, RetrieverResponse } from '../types/retrievers';
import { NestedSpanData, TraceData } from '../types/trace';
import { logger } from './logger';
import { stackTraceSpans } from './trace';

export type EvalExtractorFn = (t: TraceData) => any;

export const EVALUATOR_ACTION_PREFIX = '/evaluator';

// Update js/ai/src/evaluators.ts if you change this value
export const EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName';
export const EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition';
export const EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled';

export function evaluatorName(action: Action) {
  return `${EVALUATOR_ACTION_PREFIX}/${action.name}`;
}

export function isEvaluator(key: string) {
  return key.startsWith(EVALUATOR_ACTION_PREFIX);
}

export async function confirmLlmUse(
  evaluatorActions: Action[]
): Promise<boolean> {
  const isBilled = evaluatorActions.some(
    (action) =>
      action.metadata && action.metadata[EVALUATOR_METADATA_KEY_IS_BILLED]
  );

  if (!isBilled) {
    return true;
  }

  const answers = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirm',
      message:
        'For each example, the evaluation makes calls to APIs that may result in being charged. Do you wish to proceed?',
      default: false,
    },
  ]);

  return answers.confirm;
}

function getRootSpan(trace: TraceData): NestedSpanData | undefined {
  return stackTraceSpans(trace);
}

function safeParse(value?: string) {
  if (value) {
    try {
      return JSON.parse(value);
    } catch (e) {
      return '';
    }
  }
  return '';
}

const DEFAULT_INPUT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  const rootSpan = getRootSpan(trace);
  return safeParse(rootSpan?.attributes['genkit:input'] as string);
};
const DEFAULT_OUTPUT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  const rootSpan = getRootSpan(trace);
  return safeParse(rootSpan?.attributes['genkit:output'] as string);
};
const DEFAULT_CONTEXT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  return Object.values(trace.spans)
    .filter((s) => s.attributes['genkit:metadata:subtype'] === 'retriever')
    .flatMap((s) => {
      const output: RetrieverResponse = safeParse(
        s.attributes['genkit:output'] as string
      );
      if (!output) {
        return [];
      }
      return output.documents.flatMap((d: DocumentData) =>
        d.content.map((c) => c.text).filter((text): text is string => !!text)
      );
    });
};

const DEFAULT_FLOW_EXTRACTORS: Record<EvalField, EvalExtractorFn> = {
  input: DEFAULT_INPUT_EXTRACTOR,
  output: DEFAULT_OUTPUT_EXTRACTOR,
  context: DEFAULT_CONTEXT_EXTRACTOR,
};

const DEFAULT_MODEL_EXTRACTORS: Record<EvalField, EvalExtractorFn> = {
  input: DEFAULT_INPUT_EXTRACTOR,
  output: DEFAULT_OUTPUT_EXTRACTOR,
  context: () => [],
};

export async function getDefaultEvalExtractors(
  actionRef: string
): Promise<Record<string, EvalExtractorFn>> {
  if (actionRef.startsWith('/model')) {
    // Always use defaults for model extraction.
    logger.debug(
      'getDefaultEvalExtractors - modelRef provided, using default extractors'
    );
    return DEFAULT_MODEL_EXTRACTORS;
  }
  return DEFAULT_FLOW_EXTRACTORS;
}

export async function augmentCustomExtractor(
  manager: RuntimeManager,
  actionRef: string,
  input: EvalInput
): Promise<EvalInput> {
  const actions = await manager.listActions();
  const customExtractorAction = Object.values(actions).find(
    (action: Action) =>
      action.key.startsWith('/custom') &&
      action.metadata?.targetActionRef === actionRef
  );
  if (!customExtractorAction) {
    return input;
  }
  const response = await manager.runAction({
    key: customExtractorAction.key,
    input,
  });
  return response.result as EvalInput;
}

/**Global function to generate testCaseId */
export function generateTestCaseId() {
  return randomUUID();
}

/** Load a {@link Dataset} file. Supports JSON / JSONL */
export async function loadInferenceDatasetFile(
  fileName: string
): Promise<Dataset> {
  const isJsonl = fileName.endsWith('.jsonl');

  if (isJsonl) {
    return await readJsonlForInference(fileName);
  } else {
    const parsedData = JSON.parse(await readFile(fileName, 'utf8'));
    let dataset = InferenceDatasetSchema.parse(parsedData);
    dataset = dataset.map((sample: InferenceSample) => ({
      ...sample,
      testCaseId: sample.testCaseId ?? generateTestCaseId(),
    }));
    return DatasetSchema.parse(dataset);
  }
}

/** Load a {@link EvalInputDataset} file. Supports JSON / JSONL */
export async function loadEvaluationDatasetFile(
  fileName: string
): Promise<EvalInputDataset> {
  const isJsonl = fileName.endsWith('.jsonl');

  if (isJsonl) {
    return await readJsonlForEvaluation(fileName);
  } else {
    const parsedData = JSON.parse(await readFile(fileName, 'utf8'));
    let evaluationInput = EvaluationDatasetSchema.parse(parsedData);
    evaluationInput = evaluationInput.map((evalSample: EvaluationSample) => ({
      ...evalSample,
      testCaseId: evalSample.testCaseId ?? generateTestCaseId(),
      traceIds: evalSample.traceIds ?? [],
    }));
    return EvalInputDatasetSchema.parse(evaluationInput);
  }
}

async function readJsonlForInference(fileName: string): Promise<Dataset> {
  const lines = await readLines(fileName);
  const samples: Dataset = [];
  for (const line of lines) {
    const parsedSample = InferenceSampleSchema.parse(JSON.parse(line));
    samples.push({
      ...parsedSample,
      testCaseId: parsedSample.testCaseId ?? generateTestCaseId(),
    });
  }
  return samples;
}

async function readJsonlForEvaluation(
  fileName: string
): Promise<EvalInputDataset> {
  const lines = await readLines(fileName);
  const inputs: EvalInputDataset = [];
  for (const line of lines) {
    const parsedSample = EvaluationSampleSchema.parse(JSON.parse(line));
    inputs.push({
      ...parsedSample,
      testCaseId: parsedSample.testCaseId ?? generateTestCaseId(),
      traceIds: parsedSample.traceIds ?? [],
    });
  }
  return inputs;
}

async function readLines(fileName: string): Promise<string[]> {
  const lines: string[] = [];
  const fileStream = createReadStream(fileName);
  const rl = createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    lines.push(line);
  }
  return lines;
}
