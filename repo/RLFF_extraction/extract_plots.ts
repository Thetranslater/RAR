import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs"
import { createHash } from "node:crypto"
import { basename, dirname, extname, resolve } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
import { setTimeout as delay } from "node:timers/promises"
import { create_llm, type ModelType } from "../../LLM/create_llm.ts"
import type { BaseMessage } from "../../LLM/types.ts"

const JSON_FENCE_RE = /^```(?:json)?\s*([\s\S]*?)\s*```$/i
const PROMPT_SEPARATOR_RE = /^----------$/gm
const PROMPT_V2_REMOVED_RULE_RE =
  /^\s*5\.\s*\[ADVISE\]剧情的开头\*可能\*出现在新自然段。\s*$/m
const SENTENCE_END_RE =
  /(?:[。！？!?]+|…{1,2}|\.{3,}|\.(?=\s|$))[”’"」』】）》〉〕〗〙〛）)\]]*|\n[ \t\u3000]*\n/g
const TRAILING_BLANK_LINES_RE = /\n[ \t\u3000]*\n$/
const RETRY_FIX_ONLY_INSTRUCTION =
  "Please only fix the issues; do not attempt to rerun the task or generate a new result."

type JsonObject = Record<string, unknown>

export type PlotCharacter = {
  name: string[]
  description: string
}

export type PlotResult = {
  characters: PlotCharacter[]
  plots: Array<[string | null, string | null]>
  state: "truncated" | "finished"
}

export type PlotChunk = {
  volume: string | null
  chapter: string | null
  token_count: number | null
  text: string
  _runtime_chunk_id?: string
}

export type PromptParts = {
  system: string
  userPrefix: string
}

export type PromptVariant = "v1" | "v2"

export type RetryStrategy = {
  promptVariant: PromptVariant
  reasoningEffort: "max" | "high"
}

class RetryablePlotValidationError extends Error {
  readonly feedback: string

  constructor(message: string, feedback: string) {
    super(message)
    this.name = "RetryablePlotValidationError"
    this.feedback = feedback
  }
}

type PlotValidationIssue = {
  message: string
  outputLabel: string
  output: unknown
}

export type PlotJob = {
  sequenceIndex: number
  chunkId: string
  chunk: PlotChunk
  request: {
    input: string
    previous: string
    next: string
  }
}

type Args = {
  inputJson: string
  outputJsonl: string
  prompt: string
  provider: ModelType
  model: string
  baseURL?: string
  contextSentences: number
  concurrency: number
  requestsPerSecond: number
  retries: number
  retryDelayMs: number
  timeoutMs: number
  maxTokens: number
  temperature: number
  limit?: number
  overwrite: boolean
  dryRun: boolean
  debug: boolean
  stream: boolean
}

type LoadedChunks = {
  title: string
  chunks: PlotChunk[]
}

export type DebugAttemptStart = {
  attempt: number
  totalAttempts: number
  messages: BaseMessage[]
  parameters: JsonObject
  startedAt: string
}

export type LLMOutputInfo = {
  content: string | null
  finish_reason: string | null
  input_tokens: number | null
  output_tokens: number | null
  total_tokens: number | null
}

export type DebugAttemptDone = {
  attempt: number
  totalAttempts: number
  finishedAt: string
  durationMs: number
  thinking: string | null
  output: LLMOutputInfo
  status:
  | "success"
  | "length"
  | "validation_error"
  | "response_error"
  | "request_error"
  parsedJson?: JsonObject
  validatedResult?: PlotResult
  error?: string
}

export type DebugStreamKind = "thinking" | "content"

type DebugAttemptLogger = {
  start: (attempt: DebugAttemptStart) => void
  stream: (kind: DebugStreamKind, content: string) => void
  done: (attempt: DebugAttemptDone) => void
}

function projectRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../..")
}

function safeFilenamePart(value: string): string {
  const sanitized = value
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/[ .]+$/g, "")
  return sanitized || "chunk"
}

export function debugLogPath(
  inputJson: string,
  chunkId: string,
  root = projectRoot(),
): string {
  const inputName = basename(inputJson, extname(inputJson))
  return resolve(
    root,
    "debug",
    "plots",
    safeFilenamePart(inputName),
    `${safeFilenamePart(chunkId)}-log.txt`,
  )
}

export function formatDebugAttemptStart(attempt: DebugAttemptStart): string {
  return [
    `=== ATTEMPT ${attempt.attempt}/${attempt.totalAttempts} ===`,
    `started_at: ${attempt.startedAt}`,
    "",
    "--- MESSAGES ---",
    JSON.stringify(attempt.messages, null, 2),
    "",
    "--- ATTEMPT PARAMETERS ---",
    JSON.stringify(attempt.parameters, null, 2),
    "",
    "",
  ].join("\n")
}

export function formatDebugAttemptDone(
  attempt: DebugAttemptDone,
  includeResponse = true,
): string {
  const lines: string[] = []
  if (includeResponse) {
    const { content, ...metadata } = attempt.output
    lines.push(
      "--- THINKING ---",
      attempt.thinking ?? "(no thinking content returned)",
      "",
      "--- OUTPUT ---",
      "content:",
      content ?? "(no assistant output text returned)",
      "",
      "metadata:",
      JSON.stringify(metadata, null, 2),
      "",
    )
  }
  lines.push(
    `=== ATTEMPT ${attempt.attempt}/${attempt.totalAttempts} DONE ===`,
    `finished_at: ${attempt.finishedAt}`,
    `duration_ms: ${attempt.durationMs}`,
    "",
    "--- PROCESSING RESULT ---",
    `status: ${attempt.status}`,
  )
  if (attempt.error !== undefined) lines.push(`error: ${attempt.error}`)
  if (attempt.parsedJson !== undefined) {
    lines.push("parsed_json:", JSON.stringify(attempt.parsedJson, null, 2))
  }
  if (attempt.validatedResult !== undefined) {
    lines.push("validated_result:", JSON.stringify(attempt.validatedResult, null, 2))
  }
  return `${lines.join("\n")}\n\n`
}

function initializeDebugLog(
  path: string,
  sourceTitle: string,
  job: PlotJob,
  callParameters: JsonObject,
): void {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(
    path,
    [
      "RLFF plot extraction debug log",
      "",
      "=== CHUNK ===",
      `source_title: ${sourceTitle}`,
      `sequence_index: ${job.sequenceIndex}`,
      `chunk_id: ${job.chunkId}`,
      `volume: ${job.chunk.volume ?? ""}`,
      `chapter: ${job.chunk.chapter ?? ""}`,
      `token_count: ${job.chunk.token_count ?? ""}`,
      "",
      "=== CALL PARAMETERS ===",
      JSON.stringify(callParameters, null, 2),
      "",
      "",
    ].join("\n"),
    "utf8",
  )
}

function createDebugAttemptLogger(
  path: string,
  streamed: boolean,
): DebugAttemptLogger {
  let currentSection: DebugStreamKind | null = null
  const write = (text: string): void => appendFileSync(path, text, "utf8")

  return {
    start(attempt): void {
      currentSection = null
      write(formatDebugAttemptStart(attempt))
    },
    stream(kind, content): void {
      if (!content) return
      if (currentSection !== kind) {
        const heading =
          kind === "thinking"
            ? "--- THINKING (STREAM) ---"
            : "--- OUTPUT (STREAM) ---\ncontent:"
        const separator = currentSection === null ? "" : "\n\n"
        write(`${separator}${heading}\n`)
        currentSection = kind
      }
      write(content)
    },
    done(attempt): void {
      if (streamed) {
        if (currentSection !== "content") {
          write(`${currentSection === null ? "" : "\n\n"}--- OUTPUT ---`)
        }
        const { content: _, ...metadata } = attempt.output
        write(`\n\nmetadata:\n${JSON.stringify(metadata, null, 2)}\n\n`)
      }
      write(formatDebugAttemptDone(attempt, !streamed))
    }
  }
}

function loadEnvFile(path: string): void {
  if (!existsSync(path)) return
  const text = readFileSync(path, "utf8").replace(/^\uFEFF/, "")
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue
    const separator = line.indexOf("=")
    if (separator <= 0) continue
    const key = line.slice(0, separator).trim()
    let value = line.slice(separator + 1).trim()
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1)
    }
    if (process.env[key] === undefined) process.env[key] = value
  }
}

export function splitPrompt(prompt: string, contextSentences: number): PromptParts {
  const normalized = prompt.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
  const delimiter = [...normalized.matchAll(PROMPT_SEPARATOR_RE)].at(-1)
  if (delimiter?.index === undefined) throw new Error("Prompt has no standalone ---------- separator.")
  const substitute = (value: string): string =>
    value.trim().replaceAll("{k}", String(contextSentences))
  const system = substitute(normalized.slice(0, delimiter.index))
  const userPrefix = substitute(
    normalized.slice(delimiter.index + delimiter[0].length),
  )
  if (!system) throw new Error("Prompt system section is empty.")
  return { system, userPrefix }
}

export function createPromptV2(prompt: PromptParts): PromptParts {
  const system = prompt.system.replace(PROMPT_V2_REMOVED_RULE_RE, "").trim()
  if (system === prompt.system) {
    throw new Error("Cannot create prompt v2: removable rule not found.")
  }
  return { system, userPrefix: prompt.userPrefix }
}

export function nextLengthRetryStrategy(
  strategy: RetryStrategy,
): RetryStrategy | null {
  if (strategy.promptVariant === "v1") {
    return { promptVariant: "v2", reasoningEffort: "max" }
  }
  if (strategy.reasoningEffort === "max") {
    return { promptVariant: "v2", reasoningEffort: "high" }
  }
  return null
}

export function isTimeoutError(error: unknown): boolean {
  if (error instanceof Error && error.name === "AbortError") return true
  const message = error instanceof Error ? error.message : String(error)
  return /\btime(?:d\s*out|out)\b/i.test(message)
}

function isLengthFinishReason(reason: string | null): boolean {
  return reason !== null &&
    ["length", "max_tokens", "max_output_tokens"].includes(reason.toLowerCase())
}

export function loadChunks(path: string): LoadedChunks {
  const root = JSON.parse(
    readFileSync(path, "utf8").replace(/^\uFEFF/, ""),
  ) as { title?: string; chunks?: PlotChunk[] }
  if (!Array.isArray(root.chunks)) throw new Error("Missing chunks array.")
  const chunks = root.chunks.map((chunk, index): PlotChunk => {
    if (typeof chunk?.text !== "string" || !chunk.text.trim()) {
      throw new Error(`chunks[${index}].text must be a non-empty string.`)
    }
    return {
      volume: chunk.volume ?? null,
      chapter: chunk.chapter ?? null,
      token_count: chunk.token_count ?? null,
      text: chunk.text,
      _runtime_chunk_id: `chunk-${String(index).padStart(6, "0")}`,
    }
  })
  return {
    title: typeof root.title === "string" ? root.title : basename(path, ".json"),
    chunks,
  }
}

export function splitSentences(text: string): string[] {
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim()
  if (!normalized) return []

  const sentences: string[] = []
  let start = 0
  for (const match of normalized.matchAll(SENTENCE_END_RE)) {
    const end = match.index + match[0].length
    const segment = normalized
      .slice(start, end)
      .replace(TRAILING_BLANK_LINES_RE, "")
      .trim()
    if (segment) sentences.push(segment)
    start = end
  }
  const tail = normalized.slice(start).trim()
  if (tail) sentences.push(tail)
  return sentences
}

function sentenceContext(text: string, count: number, fromEnd: boolean): string {
  if (count === 0) return ""
  const sentences = splitSentences(text)
  return (fromEnd ? sentences.slice(-count) : sentences.slice(0, count)).join("\n")
}

function sameSection(left: PlotChunk, right: PlotChunk): boolean {
  return left.volume === right.volume && left.chapter === right.chapter
}

export function buildJobs(
  chunks: PlotChunk[],
  contextSentences: number,
): PlotJob[] {
  if (!Number.isInteger(contextSentences) || contextSentences < 0) {
    throw new Error("contextSentences must be a non-negative integer.")
  }
  return chunks.map((chunk, index): PlotJob => {
    const before = chunks[index - 1]
    const after = chunks[index + 1]
    const previous = before && sameSection(before, chunk)
      ? sentenceContext(before.text, contextSentences, true) : ""
    const next = after && sameSection(after, chunk)
      ? sentenceContext(after.text, contextSentences, false) : ""
    return {
      sequenceIndex: index,
      chunkId: chunk._runtime_chunk_id ?? `chunk-${String(index).padStart(6, "0")}`,
      chunk,
      request: { input: chunk.text, previous, next },
    }
  })
}

export function buildMessages(
  prompt: PromptParts,
  request: PlotJob["request"],
  retryFeedback?: string,
  retryOutput?: string,
): BaseMessage[] {
  const inputJson = JSON.stringify(request, null, 2)
  const userContent = prompt.userPrefix
    ? `${prompt.userPrefix}\n${inputJson}`
    : inputJson
  const messages: BaseMessage[] = [
    { role: "system", content: prompt.system },
    { role: "user", content: userContent },
  ]
  if (retryFeedback !== undefined) {
    if (retryOutput !== undefined) {
      messages.push({ role: "ai", content: retryOutput })
    }
    messages.push({ role: "user", content: retryFeedback })
  }
  return messages
}

export function parseJsonObject(text: string): JsonObject {
  let source = text.trim()
  const fenced = JSON_FENCE_RE.exec(source)
  if (fenced) source = fenced[1].trim()
  const parsed = JSON.parse(source)
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("LLM output is not a JSON object.")
  }
  return parsed as JsonObject
}

function retryableValidationError(
  issues: PlotValidationIssue[],
): RetryablePlotValidationError {
  const messages = issues.map((issue) => `[Error]${issue.message}`)
  const feedback = issues.map((issue) => [
    `[Error]${issue.message}`,
    RETRY_FIX_ONLY_INSTRUCTION,
    "",
    `${issue.outputLabel}:`,
    JSON.stringify(issue.output, null, 2),
  ].join("\n")).join("\n\n")
  return new RetryablePlotValidationError(messages.join("\n"), feedback)
}

function invalidJsonValidationError(
  output: string,
  error: unknown,
): RetryablePlotValidationError {
  const detail = error instanceof Error ? error.message : String(error)
  const message = `Result is not valid JSON: ${detail}`
  return new RetryablePlotValidationError(
    `[Error]${message}`,
    [
      `[Error]${message}`,
      RETRY_FIX_ONLY_INSTRUCTION,
      "",
      "Output is:",
      output,
    ].join("\n"),
  )
}

export function validateResult(raw: JsonObject, inputText: string): PlotResult {
  const issues: PlotValidationIssue[] = []
  const fields = Object.keys(raw).sort()
  if (fields.join(",") !== "characters,plots,state") {
    issues.push({
      message: "Output fields must be exactly: characters, plots, state.",
      outputLabel: "Output",
      output: raw,
    })
  }
  const rawCharacters = Array.isArray(raw.characters) ? raw.characters : null
  if (rawCharacters === null) {
    issues.push({
      message: "characters must be an array.",
      outputLabel: "Output characters",
      output: raw.characters,
    })
  }
  const rawPlots = Array.isArray(raw.plots) ? raw.plots : null
  if (rawPlots === null) {
    issues.push({
      message: "plots must be an array.",
      outputLabel: "Output plots",
      output: raw.plots,
    })
  }
  if (raw.state !== "truncated" && raw.state !== "finished") {
    issues.push({
      message: "state must be truncated or finished.",
      outputLabel: "Output state",
      output: raw.state,
    })
  }

  const characters: PlotCharacter[] = []
  rawCharacters?.forEach((value, index) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      issues.push({
        message: `characters[${index}] must be an object.`,
        outputLabel: `Output characters[${index}] is`,
        output: value,
      })
      return
    }
    const character = value as JsonObject
    let validCharacter = true
    if (
      Object.keys(character).sort().join(",") !== "description,name"
    ) {
      issues.push({
        message: `characters[${index}] fields must be exactly: name, description.`,
        outputLabel: `Output characters[${index}] is`,
        output: character,
      })
      validCharacter = false
    }
    if (
      !Array.isArray(character.name) ||
      character.name.length === 0 ||
      character.name.some(
        (name) => typeof name !== "string" || !name.trim(),
      )
    ) {
      issues.push({
        message: `characters[${index}].name must be a non-empty string array.`,
        outputLabel: `Output characters[${index}] is`,
        output: character,
      })
      validCharacter = false
    }
    if (
      typeof character.description !== "string" ||
      !character.description.trim()
    ) {
      issues.push({
        message: `characters[${index}].description must be a non-empty string.`,
        outputLabel: `Output characters[${index}] is`,
        output: character,
      })
      validCharacter = false
    }
    if (validCharacter) {
      characters.push({
        name: character.name as string[],
        description: character.description as string,
      })
    }
  })

  const plots: Array<[string | null, string | null]> = []
  rawPlots?.forEach((plot, index) => {
    if (
      !Array.isArray(plot) ||
      plot.length !== 2 ||
      !plot.every(
        (boundary) => boundary === null || typeof boundary === "string",
      )
    ) {
      issues.push({
        message: `plots[${index}] must contain exactly two string-or-null boundaries.`,
        outputLabel: `Output plots[${index}] is`,
        output: plot,
      })
      return
    }
    const [rawStart, rawEnd] = plot as [string | null, string | null]
    const start = rawStart === "null" ? null : rawStart
    const end = rawEnd === "null" ? null : rawEnd
    if (start === null && end === null) {
      if (rawPlots.length !== 1) {
        issues.push({
          message: `plots[${index}] with two null boundaries must be the only plot.`,
          outputLabel: "Output plots is",
          output: rawPlots,
        })
      }
      plots.push([start, end])
      return
    }
    for (const [name, boundary] of [["start", start], ["end", end]] as const) {
      if (boundary === "") {
        issues.push({
          message: `plots[${index}].${name} must use null instead of an empty string.`,
          outputLabel: `Output plots[${index}] is`,
          output: plot,
        })
      } else if (boundary !== null && !inputText.includes(boundary)) {
        issues.push({
          message: `plots[${index}].${name} is not an exact substring of input.`,
          outputLabel: `Output plots[${index}] is`,
          output: plot,
        })
      }
    }
    plots.push([start, end])
  })

  if (issues.length > 0) throw retryableValidationError(issues)

  const normalizedPlots =
    plots.length === 1 && plots[0][0] === null && plots[0][1] === null
      ? []
      : plots
  return {
    characters,
    plots: normalizedPlots,
    state: raw.state as "truncated" | "finished",
  }
}

function assistantText(response: any): string {
  const content = [
    response?.choices?.[0]?.message?.content,
    response?.output?.choices?.[0]?.message?.content,
    response?.output?.text,
    response?.content,
  ].find((value) => typeof value === "string")
  if (typeof content !== "string") throw new Error("LLM response contains no assistant text.")
  return content
}

export function assistantThinking(response: any): string {
  const content = [
    response?.choices?.[0]?.message?.reasoning_content,
    response?.choices?.[0]?.message?.thinking,
    response?.output?.choices?.[0]?.message?.reasoning_content,
    response?.output?.choices?.[0]?.message?.thinking,
    response?.output?.message?.reasoning_content,
    response?.output?.message?.thinking,
    response?.reasoning_content,
    response?.thinking,
  ].find((value) => typeof value === "string")
  return typeof content === "string" ? content.trim() : ""
}

function finiteNumber(...values: unknown[]): number | null {
  const value = values.find((item) => typeof item === "number" && Number.isFinite(item))
  return typeof value === "number" ? value : null
}

function firstString(...values: unknown[]): string | null {
  const value = values.find((item) => typeof item === "string")
  return typeof value === "string" ? value : null
}

export function extractLLMOutputInfo(
  response: any,
  content: string | null,
): LLMOutputInfo {
  const usage =
    response?.usage ?? response?.output?.usage ?? response?.args?.usage ?? {}
  return {
    content,
    finish_reason: firstString(
      response?.choices?.[0]?.finish_reason,
      response?.output?.choices?.[0]?.finish_reason,
      response?.finish_reason,
      response?.stop_reason,
      response?.args?.finish_reason,
      response?.args?.stop_reason,
    ),
    input_tokens: finiteNumber(
      usage.input_tokens,
      usage.prompt_tokens,
      response?.input_tokens,
      response?.prompt_tokens,
    ),
    output_tokens: finiteNumber(
      usage.output_tokens,
      usage.completion_tokens,
      response?.output_tokens,
      response?.completion_tokens,
    ),
    total_tokens: finiteNumber(usage.total_tokens, response?.total_tokens),
  }
}

export async function collectAssistantStream(
  source: AsyncIterable<unknown>,
  onChunk?: (kind: DebugStreamKind, content: string) => void,
): Promise<{
  thinking: string
  content: string
  output: LLMOutputInfo
}> {
  let thinking = ""
  let content = ""
  const metadata: JsonObject = {}
  for await (const chunk of source) {
    if (!chunk || typeof chunk !== "object") continue
    const event = chunk as Record<string, unknown>
    if (event.args && typeof event.args === "object") {
      const args = event.args as JsonObject
      Object.assign(metadata, args)
      if (args.usage && typeof args.usage === "object") {
        metadata.usage = { ...(metadata.usage as JsonObject), ...args.usage as JsonObject }
      }
    }
    if (
      (event.type !== "thinking" && event.type !== "content") ||
      typeof event.content !== "string" || !event.content
    ) {
      continue
    }
    if (event.type === "thinking") {
      thinking += event.content
      onChunk?.("thinking", event.content)
    } else {
      content += event.content
      onChunk?.("content", event.content)
    }
  }
  return {
    thinking: thinking.trim(),
    content,
    output: extractLLMOutputInfo(metadata, content || null),
  }
}

class RateGate {
  readonly intervalMs: number
  nextStart = 0
  tail: Promise<void> = Promise.resolve()

  constructor(requestsPerSecond: number) {
    this.intervalMs = 1000 / requestsPerSecond
  }

  async wait(): Promise<void> {
    let release = (): void => { }
    const turn = new Promise<void>((resolveTurn) => {
      release = resolveTurn
    })
    const previous = this.tail
    this.tail = turn
    await previous
    try {
      const waitMs = Math.max(0, this.nextStart - Date.now())
      if (waitMs > 0) await delay(waitMs)
      this.nextStart = Date.now() + this.intervalMs
    } finally {
      release()
    }
  }
}

async function extractOne(
  llm: ReturnType<typeof create_llm>,
  limiter: RateGate,
  job: PlotJob,
  promptV1: PromptParts,
  promptV2: PromptParts,
  retries: number,
  retryDelayMs: number,
  streamed: boolean,
  provider: ModelType,
  baseRuntimeParameters: JsonObject,
  debugLogger?: DebugAttemptLogger,
): Promise<PlotResult> {
  const totalAttempts = retries + 1
  let strategy: RetryStrategy = { promptVariant: "v1", reasoningEffort: "max" }
  let retryFeedback: string | undefined
  let retryOutput: string | undefined

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const attemptNumber = attempt + 1
    const activePrompt = strategy.promptVariant === "v1" ? promptV1 : promptV2
    const messages = buildMessages(
      activePrompt,
      job.request,
      retryFeedback,
      retryOutput,
    )
    const runtimeParameters: JsonObject = {
      ...baseRuntimeParameters,
      ...(provider === "DeepSeek"
        ? { reasoning_effort: strategy.reasoningEffort }
        : {}),
    }
    await limiter.wait()
    const startedAt = new Date()
    debugLogger?.start({
      attempt: attemptNumber,
      totalAttempts,
      messages,
      parameters: {
        prompt_variant: strategy.promptVariant,
        stream: streamed,
        ...runtimeParameters,
      },
      startedAt: startedAt.toISOString(),
    })

    let received = false
    let thinking = ""
    let content = ""
    let output = extractLLMOutputInfo(null, null)
    const finish = (
      status: DebugAttemptDone["status"],
      error?: unknown,
      parsedJson?: JsonObject,
      validatedResult?: PlotResult,
    ): void => {
      const finishedAt = new Date()
      debugLogger?.done({
        attempt: attemptNumber,
        totalAttempts,
        finishedAt: finishedAt.toISOString(),
        durationMs: finishedAt.getTime() - startedAt.getTime(),
        thinking: thinking || null,
        output,
        status,
        ...(error === undefined
          ? {}
          : { error: error instanceof Error ? error.message : String(error) }),
        ...(parsedJson === undefined ? {} : { parsedJson }),
        ...(validatedResult === undefined ? {} : { validatedResult }),
      })
    }

    try {
      if (streamed) {
        const responseStream = await llm.stream(messages, runtimeParameters as any)
        received = true
        const collected = await collectAssistantStream(
          responseStream,
          debugLogger ? (kind, chunk) => debugLogger.stream(kind, chunk) : undefined,
        )
          ; ({ thinking, content, output } = collected)
        if (!content) throw new Error("LLM response contains no assistant text.")
      } else {
        const response = await llm.invoke(messages, runtimeParameters as any)
        received = true
        thinking = assistantThinking(response)
        output = extractLLMOutputInfo(response, null)
        content = assistantText(response)
        output.content = content
      }
    } catch (error) {
      finish(received ? "response_error" : "request_error", error)
      if (isTimeoutError(error) && attempt < retries) {
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }

    if (isLengthFinishReason(output.finish_reason)) {
      const error = new Error(`LLM stopped because finish_reason=${output.finish_reason}.`)
      finish("length", error)
      const nextStrategy = nextLengthRetryStrategy(strategy)
      if (nextStrategy && attempt < retries) {
        strategy = nextStrategy
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }

    let parsedJson: JsonObject
    try {
      parsedJson = parseJsonObject(content)
    } catch (error) {
      const retryError = invalidJsonValidationError(content, error)
      finish("validation_error", retryError)
      if (attempt < retries) {
        retryFeedback = retryError.feedback
        retryOutput = content
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw retryError
    }

    try {
      const result = validateResult(parsedJson, job.request.input)
      finish("success", undefined, parsedJson, result)
      return result
    } catch (error) {
      finish("validation_error", error, parsedJson)
      if (error instanceof RetryablePlotValidationError && attempt < retries) {
        retryFeedback = error.feedback
        retryOutput = content
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }
  }
  throw new Error("Plot extraction retry loop exhausted.")
}

function completedChunkIds(path: string): Set<string> {
  if (!existsSync(path)) return new Set()
  return new Set(
    readFileSync(path, "utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line).chunk_id)
      .filter((id): id is string => typeof id === "string"),
  )
}

export function outputRecord(
  sourceTitle: string,
  job: PlotJob,
  result: PlotResult,
  model: string,
): JsonObject {
  return {
    model,
    source_title: sourceTitle,
    chunk_id: job.chunkId,
    volume: job.chunk.volume,
    chapter: job.chunk.chapter,
    result,
  }
}

function valueAfter(argv: string[], index: number, option: string): string {
  const value = argv[index + 1]
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${option}`)
  return value
}

function numericValue(value: string, option: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) throw new Error(`Invalid number for ${option}: ${value}`)
  return parsed
}

function providerValue(value: string): ModelType {
  const normalized = value.toLowerCase()
  if (normalized === "deepseek") return "DeepSeek"
  if (normalized === "glm") return "GLM"
  if (normalized === "qwen") return "Qwen"
  throw new Error(`Unsupported provider: ${value}`)
}

export function parseArgs(argv: string[]): Args {
  const positional: string[] = []
  const configuredModel =
    process.env.RLFF_PLOT_MODEL ?? process.env.RLFF_VERIFIER_MODEL
  const args: Omit<Args, "inputJson" | "outputJsonl"> = {
    prompt: resolve(projectRoot(), "plot_extraction.txt"),
    provider: providerValue(process.env.RLFF_PLOT_PROVIDER ?? "DeepSeek"),
    model: configuredModel ?? "",
    baseURL: undefined,
    contextSentences: 3,
    concurrency: 1,
    requestsPerSecond: 5,
    retries: 2,
    retryDelayMs: 2000,
    timeoutMs: 120_000,
    maxTokens: 8 * 1024,
    temperature: 0,
    limit: undefined,
    overwrite: false,
    dryRun: false,
    debug: false,
    stream: false,
  }
  const numericOptions: Record<string, keyof typeof args> = {
    "--context-sentences": "contextSentences",
    "--concurrency": "concurrency",
    "--requests-per-second": "requestsPerSecond",
    "--retries": "retries",
    "--retry-delay-ms": "retryDelayMs",
    "--timeout-ms": "timeoutMs",
    "--max-tokens": "maxTokens",
    "--temperature": "temperature",
    "--limit": "limit",
  }

  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index]
    if (!option.startsWith("--")) {
      positional.push(option)
      continue
    }
    if (option === "--overwrite") args.overwrite = true
    else if (option === "--dry-run") args.dryRun = true
    else if (option === "--debug") args.debug = true
    else if (option === "--stream") args.stream = true
    else {
      const value = valueAfter(argv, index, option)
      if (option === "--prompt") args.prompt = value
      else if (option === "--provider") args.provider = providerValue(value)
      else if (option === "--model") args.model = value
      else if (option === "--base-url") args.baseURL = value
      else if (numericOptions[option]) {
        ; (args as any)[numericOptions[option]] = numericValue(value, option)
      } else throw new Error(`Unknown option: ${option}`)
      index += 1
    }
  }

  if (positional.length !== 2) {
    throw new Error(
      "Usage: extract_plots.ts <input_json> <output_jsonl> [options]",
    )
  }
  for (const [name, value, minimum] of [
    ["contextSentences", args.contextSentences, 0],
    ["concurrency", args.concurrency, 1],
    ["retries", args.retries, 0],
    ["retryDelayMs", args.retryDelayMs, 0],
    ["timeoutMs", args.timeoutMs, 1],
    ["maxTokens", args.maxTokens, 1],
  ] as const) {
    if (!Number.isInteger(value) || value < minimum) {
      throw new Error(`${name} has an invalid value: ${value}`)
    }
  }
  if (args.requestsPerSecond <= 0) throw new Error("requestsPerSecond must be positive.")
  if (args.temperature < 0 || args.temperature > 2) throw new Error("temperature must be between 0 and 2.")
  if (args.limit !== undefined && (!Number.isInteger(args.limit) || args.limit < 1)) {
    throw new Error("limit must be a positive integer.")
  }
  if (!args.model) {
    if (args.provider === "DeepSeek") {
      args.model = "deepseek-v4-pro"
    } else {
      throw new Error("--model or RLFF_PLOT_MODEL is required for GLM and Qwen.")
    }
  }
  args.baseURL ??= {
    DeepSeek: process.env.DEEPSEEK_BASE_URL,
    GLM: process.env.GLM_BASE_URL,
    Qwen: process.env.DASHSCOPE_BASE_URL,
  }[args.provider]

  return {
    ...args,
    inputJson: positional[0],
    outputJsonl: positional[1],
  }
}

async function run(args: Args): Promise<void> {
  const loaded = loadChunks(args.inputJson)
  const promptText = readFileSync(args.prompt, "utf8").replace(/^\uFEFF/, "")
  const promptV1 = splitPrompt(promptText, args.contextSentences)
  const promptV2 = createPromptV2(promptV1)
  const promptV1Sha256 = createHash("sha256")
    .update(promptV1.system)
    .digest("hex")
  const promptV2Sha256 = createHash("sha256")
    .update(promptV2.system)
    .digest("hex")
  let jobs = buildJobs(loaded.chunks, args.contextSentences)

  const existing = args.overwrite ? new Set<string>() : completedChunkIds(args.outputJsonl)
  jobs = jobs.filter((job) => !existing.has(job.chunkId))
  if (args.limit !== undefined) jobs = jobs.slice(0, args.limit)

  if (args.overwrite && !args.dryRun && existsSync(args.outputJsonl)) {
    unlinkSync(args.outputJsonl)
  }

  console.log(JSON.stringify({
    provider: args.provider,
    model: args.model,
    source_title: loaded.title,
    chunk_count: loaded.chunks.length,
    selected_count: jobs.length,
    existing_count: existing.size,
    context_sentences: args.contextSentences,
    concurrency: args.concurrency,
    debug: args.debug,
    stream: args.stream,
    prompt_v1_sha256: promptV1Sha256,
    prompt_v2_sha256: promptV2Sha256,
  }, null, 2))

  if (args.dryRun) {
    if (jobs.length > 0) {
      console.log(JSON.stringify(buildMessages(promptV1, jobs[0].request), null, 2))
    }
    return
  }
  if (jobs.length === 0) return

  const providerParameters: JsonObject =
    args.provider === "DeepSeek" ? { reasoning_effort: "max" }
      : { enable_thinking: true }
  const clientConfig: any = {
    model: args.model,
    temperature: args.temperature,
    max_tokens: args.maxTokens,
    timeout: args.timeoutMs,
    ...(args.baseURL ? { baseURL: args.baseURL } : {}),
    ...providerParameters,
  }
  const llm = create_llm(args.provider, clientConfig)
  const limiter = new RateGate(args.requestsPerSecond)
  mkdirSync(dirname(resolve(args.outputJsonl)), { recursive: true })
  const debugCallParameters: JsonObject = {
    provider: args.provider,
    model: args.model,
    max_tokens: args.maxTokens,
    temperature: args.temperature,
    timeout_ms: args.timeoutMs,
    context_sentences: args.contextSentences,
    retries: args.retries,
    retry_delay_ms: args.retryDelayMs,
    requests_per_second: args.requestsPerSecond,
    concurrency: args.concurrency,
    stream: args.stream,
    ...(args.baseURL ? { base_url: args.baseURL } : {}),
    ...providerParameters,
  }

  let cursor = 0
  let completed = 0
  const failures: Array<{ chunk_id: string; error: string }> = []

  async function worker(): Promise<void> {
    while (true) {
      const jobIndex = cursor++
      if (jobIndex >= jobs.length) return
      const job = jobs[jobIndex]
      const logPath = args.debug
        ? debugLogPath(args.inputJson, job.chunkId)
        : undefined
      if (logPath) {
        initializeDebugLog(logPath, loaded.title, job, debugCallParameters)
      }
      try {
        const result = await extractOne(
          llm,
          limiter,
          job,
          promptV1,
          promptV2,
          args.retries,
          args.retryDelayMs,
          args.stream,
          args.provider,
          clientConfig,
          logPath
            ? createDebugAttemptLogger(logPath, args.stream)
            : undefined,
        )
        const record = outputRecord(loaded.title, job, result, args.model)
        appendFileSync(args.outputJsonl, JSON.stringify(record) + "\n", "utf8")
        completed += 1
        console.log(`[${completed}/${jobs.length}] ${job.chunkId}`)
      } catch (error) {
        failures.push({
          chunk_id: job.chunkId,
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(args.concurrency, jobs.length) }, worker),
  )
  console.log(JSON.stringify({ completed, failed: failures.length, failures }, null, 2))
  if (failures.length > 0) {
    throw new Error(`${failures.length} plot extraction job(s) failed.`)
  }
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  loadEnvFile(resolve(projectRoot(), ".env"))
  await run(parseArgs(argv))
}

const entryUrl = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : ""
if (import.meta.url === entryUrl) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error))
    process.exitCode = 1
  })
}
