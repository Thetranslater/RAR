import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs"
import { basename, dirname, extname, resolve } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
import { setTimeout as delay } from "node:timers/promises"
import { create_llm, type ModelType } from "../../LLM/create_llm.ts"
import type { BaseMessage } from "../../LLM/types.ts"
import {
  assistantThinking,
  collectAssistantStream,
  extractLLMOutputInfo,
  isTimeoutError,
  type DebugStreamKind,
  type LLMOutputInfo,
} from "./extract_plots.ts"

export type JsonObject = Record<string, unknown>
type StateValue = string | number | boolean | null
export const MIN_PLOT_TOKENS = 512
const JSON_FENCE_RE = /^```(?:json)?\s*([\s\S]*?)\s*```$/i

export type StateNode = {
  name: string
  before: StateValue
  after: StateValue
  type: "initialization" | "instant"
  source: string
  description: string
}

export type StateResult = {
  nodes: StateNode[]
}

export type PromptParts = {
  system: string
  userPrefix: string
}

export type CharacterInfo = {
  name: string[]
  description: string
}

export type StateJob = {
  plotIndex: number
  chunkIndex: number
  jobId: string
  volume: string | null
  chapter: string | null
  characters: CharacterInfo[]
  text: string
}

export type LoadedJobs = {
  title: string
  jobs: StateJob[]
  allJobs: StateJob[]
  plotCount: number
  skippedPlotCount: number
}

type Args = {
  inputJson: string
  outputJsonl: string
  prompt: string
  provider: ModelType
  model: string
  baseURL?: string
  concurrency: number
  requestsPerSecond: number
  retries: number
  retryDelayMs: number
  timeoutMs: number
  maxTokens: number
  temperature: number
  start: number
  limit?: number
  overwrite: boolean
  dryRun: boolean
  debug: boolean
  stream: boolean
}

type AttemptStatus =
  | "success"
  | "length"
  | "validation_error"
  | "response_error"
  | "request_error"

class RetryableResultError extends Error {
  readonly feedback: string
  readonly nodeNames: string[]

  constructor(feedback: string, nodeNames: string[] = []) {
    super(feedback)
    this.name = "RetryableResultError"
    this.feedback = feedback
    this.nodeNames = nodeNames
  }
}

type StateValidationIssue = {
  feedback: string
  nodeName?: string
  includeInput?: boolean
}

type DebugAttemptDone = {
  attempt: number
  totalAttempts: number
  finishedAt: string
  durationMs: number
  thinking: string | null
  output: LLMOutputInfo
  status: AttemptStatus
  error?: string
  parsedJson?: JsonObject
  validatedResult?: StateResult
}

type DebugLogger = {
  start: (
    attempt: number,
    totalAttempts: number,
    startedAt: string,
    messages: BaseMessage[],
    parameters: JsonObject,
  ) => void
  stream: (kind: DebugStreamKind, content: string) => void
  done: (result: DebugAttemptDone) => void
}

const PROMPT_SEPARATOR_RE = /^----------$/gm

function projectRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../..")
}

function safeFilenamePart(value: string): string {
  return value
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/[ .]+$/g, "") || "state"
}

function jobKey(plotIndex: number, chunkIndex: number): string {
  return `${plotIndex}:${chunkIndex}`
}

export function debugLogPath(
  inputJson: string,
  jobId: string,
  root = projectRoot(),
): string {
  const inputName = basename(inputJson, extname(inputJson))
  return resolve(
    root,
    "debug",
    "states",
    safeFilenamePart(inputName),
    `${safeFilenamePart(jobId)}-log.txt`,
  )
}

function initializeDebugLog(
  path: string,
  title: string,
  job: StateJob,
  parameters: JsonObject,
): void {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, [
    "RLFF state extraction debug log",
    "",
    "=== CHUNK ===",
    `source_title: ${title}`,
    `plot_index: ${job.plotIndex}`,
    `chunk_index: ${job.chunkIndex}`,
    `job_id: ${job.jobId}`,
    `volume: ${job.volume ?? ""}`,
    `chapter: ${job.chapter ?? ""}`,
    "",
    "=== CALL PARAMETERS ===",
    JSON.stringify(parameters, null, 2),
    "",
    "",
  ].join("\n"), "utf8")
}

function createDebugLogger(path: string, streamed: boolean): DebugLogger {
  let currentSection: DebugStreamKind | null = null
  const write = (text: string): void => appendFileSync(path, text, "utf8")

  return {
    start(attempt, totalAttempts, startedAt, messages, parameters): void {
      currentSection = null
      write([
        `=== ATTEMPT ${attempt}/${totalAttempts} ===`,
        `started_at: ${startedAt}`,
        "",
        "--- MESSAGES ---",
        JSON.stringify(messages, null, 2),
        "",
        "--- ATTEMPT PARAMETERS ---",
        JSON.stringify(parameters, null, 2),
        "",
        "",
      ].join("\n"))
    },
    stream(kind, content): void {
      if (!content) return
      if (currentSection !== kind) {
        const separator = currentSection === null ? "" : "\n\n"
        const heading = kind === "thinking"
          ? "--- THINKING (STREAM) ---"
          : "--- OUTPUT (STREAM) ---\ncontent:"
        write(`${separator}${heading}\n`)
        currentSection = kind
      }
      write(content)
    },
    done(result): void {
      const { content, ...metadata } = result.output
      if (!streamed) {
        write([
          "--- THINKING ---",
          result.thinking ?? "(no thinking content returned)",
          "",
          "--- OUTPUT ---",
          "content:",
          content ?? "(no assistant output text returned)",
          "",
        ].join("\n"))
      } else if (currentSection !== "content") {
        write(`${currentSection === null ? "" : "\n\n"}--- OUTPUT ---\n`)
      }
      const lines = [
        "metadata:",
        JSON.stringify(metadata, null, 2),
        "",
        `=== ATTEMPT ${result.attempt}/${result.totalAttempts} DONE ===`,
        `finished_at: ${result.finishedAt}`,
        `duration_ms: ${result.durationMs}`,
        "",
        "--- PROCESSING RESULT ---",
        `status: ${result.status}`,
      ]
      if (result.error !== undefined) lines.push(`error: ${result.error}`)
      if (result.parsedJson !== undefined) {
        lines.push("parsed_json:", JSON.stringify(result.parsedJson, null, 2))
      }
      if (result.validatedResult !== undefined) {
        lines.push(
          "validated_result:",
          JSON.stringify(result.validatedResult, null, 2),
        )
      }
      write(`${streamed ? "\n\n" : ""}${lines.join("\n")}\n\n`)
    },
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

export function splitPrompt(prompt: string): PromptParts {
  const normalized = prompt.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
  const delimiter = [...normalized.matchAll(PROMPT_SEPARATOR_RE)].at(-1)
  if (delimiter?.index === undefined) {
    throw new Error("Prompt has no standalone ---------- separator.")
  }
  const system = normalized.slice(0, delimiter.index).trim()
  const userPrefix = normalized
    .slice(delimiter.index + delimiter[0].length)
    .trim()
  if (!system) throw new Error("Prompt system section is empty.")
  return { system, userPrefix }
}

function stringOrNull(value: unknown, field: string): string | null {
  if (value === null || value === undefined) return null
  if (typeof value !== "string") throw new Error(`${field} must be a string or null.`)
  return value
}

function characterInfo(value: unknown, field: string): CharacterInfo {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${field} must be an object.`)
  }
  const character = value as JsonObject
  if (
    !Array.isArray(character.name) ||
    character.name.length === 0 ||
    character.name.some((name) => typeof name !== "string" || !name.trim())
  ) {
    throw new Error(`${field}.name must be a non-empty string array.`)
  }
  if (typeof character.description !== "string" || !character.description.trim()) {
    throw new Error(`${field}.description must be a non-empty string.`)
  }
  return {
    name: [character.name[0]],
    description: character.description,
  }
}

export function buildJobs(
  root: JsonObject,
  fallbackTitle: string,
): LoadedJobs {
  if (!Array.isArray(root.plots)) throw new Error("Input JSON must contain plots.")

  const jobs: StateJob[] = []
  const allJobs: StateJob[] = []
  let skippedPlotCount = 0
  root.plots.forEach((rawPlot, plotIndex) => {
    if (!rawPlot || typeof rawPlot !== "object" || Array.isArray(rawPlot)) {
      throw new Error(`plots[${plotIndex}] must be an object.`)
    }
    const plot = rawPlot as JsonObject
    if (!Array.isArray(plot.chunks)) {
      throw new Error(`plots[${plotIndex}].chunks must be an array.`)
    }
    if (!Array.isArray(plot.characters)) {
      throw new Error(`plots[${plotIndex}].characters must be an array.`)
    }
    const characters = plot.characters.map((character, characterIndex) =>
      characterInfo(character, `plots[${plotIndex}].characters[${characterIndex}]`)
    )
    const volume = stringOrNull(plot.volume, `plots[${plotIndex}].volume`)
    const chapter = stringOrNull(plot.chapter, `plots[${plotIndex}].chapter`)
    const plotJobs: StateJob[] = []
    let plotTokens = 0
    plot.chunks.forEach((rawChunk, chunkIndex) => {
      if (!rawChunk || typeof rawChunk !== "object" || Array.isArray(rawChunk)) {
        throw new Error(`plots[${plotIndex}].chunks[${chunkIndex}] must be an object.`)
      }
      const chunk = rawChunk as JsonObject
      const text = chunk.text
      if (typeof text !== "string" || !text.trim()) {
        throw new Error(`plots[${plotIndex}].chunks[${chunkIndex}].text is invalid.`)
      }
      if (
        typeof chunk.token !== "number" ||
        !Number.isInteger(chunk.token) ||
        chunk.token < 0
      ) {
        throw new Error(`plots[${plotIndex}].chunks[${chunkIndex}].token is invalid.`)
      }
      plotTokens += chunk.token
      plotJobs.push({
        plotIndex,
        chunkIndex,
        jobId: `plot-${String(plotIndex).padStart(6, "0")}-chunk-${String(chunkIndex).padStart(4, "0")}`,
        volume,
        chapter,
        characters,
        text,
      })
    })
    allJobs.push(...plotJobs)
    if (plotTokens < MIN_PLOT_TOKENS) skippedPlotCount += 1
    else jobs.push(...plotJobs)
  })
  return {
    title: typeof root.title === "string" ? root.title : fallbackTitle,
    jobs,
    allJobs,
    plotCount: root.plots.length,
    skippedPlotCount,
  }
}

export function loadJobs(path: string): LoadedJobs {
  const root = JSON.parse(
    readFileSync(path, "utf8").replace(/^\uFEFF/, ""),
  ) as JsonObject
  return buildJobs(root, basename(path, ".json"))
}

export function buildMessages(
  prompt: PromptParts,
  input: string,
  characters: CharacterInfo[],
  prenodes: StateNode[],
  retryFeedback?: string,
  retryOutput?: string,
): BaseMessage[] {
  const inputJson = JSON.stringify({ characters, prenodes, input }, null, 2)
  const messages: BaseMessage[] = [
    { role: "system", content: prompt.system },
    {
      role: "user",
      content: prompt.userPrefix
        ? `${prompt.userPrefix}\n${inputJson}`
        : inputJson,
    },
  ]
  if (retryFeedback !== undefined) {
    if (retryOutput !== undefined) {
      messages.push({ role: "ai", content: retryOutput })
    }
    messages.push({ role: "user", content: retryFeedback })
  }
  return messages
}

function isStateValue(value: unknown): value is StateValue {
  return value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
}

function printableJsonValue(value: unknown): string {
  return JSON.stringify(value) ?? String(value)
}

const REPAIR_ONLY_INSTRUCTION =
  "请仅返回针对错误node的修复结果，不需要将整个输出重新生成。"
const FIX_ONLY_INSTRUCTION =
  "Please only fix the issues; do not attempt to rerun the task or generate a new result."

export function invalidJsonRetryFeedback(content: string): string {
  return [
    "[Error]结果是非法的json格式，请检查输出结果：",
    FIX_ONLY_INSTRUCTION,
    content,
    REPAIR_ONLY_INSTRUCTION,
  ].join("\n")
}

export function parseStateOutput(content: string): JsonObject {
  let source = content.trim()
  const fenced = JSON_FENCE_RE.exec(source)
  if (fenced) source = fenced[1].trim()
  const parsed: unknown = JSON.parse(source)
  if (Array.isArray(parsed)) return { nodes: parsed }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("LLM output must be a JSON object or nodes array.")
  }
  return parsed as JsonObject
}

export function normalizeStateName(name: string): string {
  const separator = name.indexOf(".")
  if (separator < 0) return name
  const attribute = name.slice(separator + 1)
  if (attribute !== "位置" && attribute !== "地点") return name
  return `${name.slice(0, separator)}.location`
}

function genericRetryFeedback(
  message: string,
  outputLabel: string,
  output: unknown,
): string {
  return [
    `[Error]${message}`,
    FIX_ONLY_INSTRUCTION,
    `${outputLabel}:`,
    JSON.stringify(output, null, 2),
  ].join("\n")
}

function typeRetryFeedback(index: number, value: unknown): string {
  return [
    `[Error]node[${index}].type类型错误，type类型必须是` +
    "initialization或instant。" +
    `当前为node[${index}].type=${printableJsonValue(value)}`,
    FIX_ONLY_INSTRUCTION,
  ].join("\n")
}

function sourceRetryFeedback(
  index: number,
  node: JsonObject,
): string {
  const name = typeof node.name === "string" ? node.name : ""
  const separator = name.indexOf(".")
  const attribute = separator >= 0 ? name.slice(separator + 1) : "attribute"
  return [
    `[Error]node[${index}].source检验错误，source必须是原文当中出现的内容。` +
    `请检查node[${index}].source是否符合以下几种错误情况：`,
    FIX_ONLY_INSTRUCTION,
    `1）非原文内容：node[${index}].source无法匹配任何原文文字，` +
    "可能进行了重写、生成、总结等错误操作。",
    `2）原文拼接：node[${index}].source通过原文不同位置的句子拼接而成，` +
    "其必须是原文的连续内容。" +
    `请选择最能体现node[${index}].${attribute}的source语句。`,
    "请检查结果:",
    JSON.stringify(node, null, 2),
  ].join("\n")
}

function validationError(
  issues: StateValidationIssue[],
  input: string,
): RetryableResultError {
  const feedbackParts = [issues.map((issue) => issue.feedback).join("\n\n")]
  if (issues.some((issue) => issue.includeInput)) {
    feedbackParts.push(`原文为：\n${input}`)
  }
  feedbackParts.push(REPAIR_ONLY_INSTRUCTION)
  return new RetryableResultError(
    feedbackParts.join("\n\n"),
    [...new Set(
      issues
        .map((issue) => issue.nodeName)
        .filter((name): name is string => name !== undefined),
    )],
  )
}

function nodeName(value: unknown): string | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const name = (value as JsonObject).name
  return typeof name === "string" && name.length > 0
    ? normalizeStateName(name)
    : null
}

function repairNodes(repair: JsonObject): unknown[] {
  if (Array.isArray(repair.nodes)) return repair.nodes
  if (!("nodes" in repair)) return [repair]
  return []
}

export function mergeNodeRepairs(
  original: JsonObject,
  repair: JsonObject,
  invalidNodeNames: readonly string[],
): JsonObject {
  const returnedNodes = repairNodes(repair)
  if (!Array.isArray(original.nodes)) {
    return { nodes: returnedNodes }
  }

  const originalNodes = [...original.nodes]
  const indexesByName = new Map<string, number[]>()
  originalNodes.forEach((node, index) => {
    const name = nodeName(node)
    if (name === null) return
    const indexes = indexesByName.get(name) ?? []
    indexes.push(index)
    indexesByName.set(name, indexes)
  })

  const matchedCounts = new Map<string, number>()
  const replacements: Array<{
    index: number
    name: string
    node: unknown
  }> = []
  const additions: unknown[] = []
  for (const node of returnedNodes) {
    const name = nodeName(node)
    const indexes = name === null ? undefined : indexesByName.get(name)
    const used = name === null ? 0 : matchedCounts.get(name) ?? 0
    if (name !== null && indexes !== undefined && used < indexes.length) {
      replacements.push({ index: indexes[used], name, node })
      matchedCounts.set(name, used + 1)
    } else {
      additions.push(node)
    }
  }

  const isWholeResult = replacements.length === originalNodes.length
  const invalidNames = new Set(invalidNodeNames)
  for (const replacement of replacements) {
    if (!isWholeResult || invalidNames.has(replacement.name)) {
      originalNodes[replacement.index] = replacement.node
    }
  }
  return { nodes: [...originalNodes, ...additions] }
}

export const MIN_SOURCE_MATCH_RATIO = 0.8

export interface SourceMatch {
  source: string
  similarity: number
}

export function matchSourceToInput(
  source: string,
  input: string,
  minimumSimilarity = MIN_SOURCE_MATCH_RATIO,
): SourceMatch | null {
  if (input.includes(source)) {
    return { source, similarity: 1 }
  }

  const pattern = Array.from(source)
  const text = Array.from(input)
  if (pattern.length === 0 || text.length === 0) return null

  // Semi-global Levenshtein distance: skipping an input prefix is free, so
  // the final row identifies the contiguous input span closest to source.
  const distances = Array.from(
    { length: pattern.length + 1 },
    () => new Uint32Array(text.length + 1),
  )
  for (let index = 0; index <= pattern.length; index += 1) {
    distances[index][0] = index
  }

  for (let patternIndex = 1; patternIndex <= pattern.length; patternIndex += 1) {
    for (let textIndex = 1; textIndex <= text.length; textIndex += 1) {
      const substitution =
        distances[patternIndex - 1][textIndex - 1] +
        (pattern[patternIndex - 1] === text[textIndex - 1] ? 0 : 1)
      const deletion = distances[patternIndex - 1][textIndex] + 1
      const insertion = distances[patternIndex][textIndex - 1] + 1
      distances[patternIndex][textIndex] = Math.min(
        substitution,
        deletion,
        insertion,
      )
    }
  }

  let end = 0
  let distance = distances[pattern.length][0]
  for (let textIndex = 1; textIndex <= text.length; textIndex += 1) {
    if (distances[pattern.length][textIndex] < distance) {
      distance = distances[pattern.length][textIndex]
      end = textIndex
    }
  }

  let patternIndex = pattern.length
  let textIndex = end
  while (patternIndex > 0) {
    const current = distances[patternIndex][textIndex]
    if (
      textIndex > 0 &&
      pattern[patternIndex - 1] === text[textIndex - 1] &&
      current ===
        distances[patternIndex - 1][textIndex - 1]
    ) {
      patternIndex -= 1
      textIndex -= 1
    } else if (current === distances[patternIndex - 1][textIndex] + 1) {
      patternIndex -= 1
    } else if (
      textIndex > 0 &&
      current === distances[patternIndex][textIndex - 1] + 1
    ) {
      textIndex -= 1
    } else {
      // The remaining edit is a substitution.
      patternIndex -= 1
      textIndex -= 1
    }
  }

  const matchedSource = text.slice(textIndex, end).join("")
  const similarity =
    1 - distance / Math.max(pattern.length, end - textIndex)
  if (similarity < minimumSimilarity) return null
  return { source: matchedSource, similarity }
}

export function validateResult(raw: JsonObject, input: string): StateResult {
  const issues: StateValidationIssue[] = []
  const rawNodes = Array.isArray(raw.nodes) ? raw.nodes : null
  if (Object.keys(raw).length !== 1 || rawNodes === null) {
    issues.push({
      feedback: genericRetryFeedback(
        "Output must contain only a nodes array.",
        "Output",
        raw,
      ),
    })
  }

  const nodes: StateNode[] = []
  rawNodes?.forEach((value, index) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      issues.push({
        feedback: genericRetryFeedback(
          `nodes[${index}] must be an object.`,
          `Output nodes[${index}] is`,
          value,
        ),
      })
      return
    }
    const node = value as JsonObject
    const currentName = typeof node.name === "string" && node.name
      ? normalizeStateName(node.name)
      : undefined
    let validNode = true
    const expected = ["after", "before", "description", "name", "source", "type"]
    if (Object.keys(node).sort().join(",") !== expected.join(",")) {
      issues.push({
        feedback: genericRetryFeedback(
          `nodes[${index}] fields are invalid.`,
          `Output nodes[${index}] is`,
          node,
        ),
        nodeName: currentName,
      })
      validNode = false
    }
    let name: string | null = null
    if (typeof node.name !== "string" || !node.name.trim()) {
      issues.push({
        feedback: genericRetryFeedback(
          `nodes[${index}].name must be a non-empty string.`,
          `Output nodes[${index}] is`,
          node,
        ),
      })
      validNode = false
    } else {
      name = normalizeStateName(node.name)
      if (!/^[^.]+\.[^.]+$/.test(name)) {
        issues.push({
          feedback: genericRetryFeedback(
            `nodes[${index}].name must use Entity.attribute format.`,
            `Output nodes[${index}] is`,
            node,
          ),
          nodeName: name,
        })
        validNode = false
      }
    }
    if (!isStateValue(node.before) || !isStateValue(node.after)) {
      issues.push({
        feedback: genericRetryFeedback(
          `nodes[${index}] before/after must be JSON scalar values.`,
          `Output nodes[${index}] is`,
          node,
        ),
        nodeName: currentName,
      })
      validNode = false
    }
    if (node.type !== "initialization" && node.type !== "instant") {
      issues.push({
        feedback: typeRetryFeedback(index, node.type),
        nodeName: currentName,
      })
      validNode = false
    }
    let sourceMatch: SourceMatch | null = null
    if (typeof node.source !== "string" || !node.source.trim()) {
      issues.push({
        feedback: sourceRetryFeedback(index, node),
        nodeName: currentName,
        includeInput: true,
      })
      validNode = false
    } else {
      sourceMatch = matchSourceToInput(node.source, input)
      if (!sourceMatch) {
        issues.push({
          feedback: sourceRetryFeedback(index, node),
          nodeName: currentName,
          includeInput: true,
        })
        validNode = false
      }
    }
    if (typeof node.description !== "string" || !node.description.trim()) {
      issues.push({
        feedback: genericRetryFeedback(
          `nodes[${index}].description must be a non-empty string.`,
          `Output nodes[${index}] is`,
          node,
        ),
        nodeName: currentName,
      })
      validNode = false
    }
    if (validNode && name !== null && sourceMatch !== null) {
      nodes.push({
        name,
        before: node.before as StateValue,
        after: node.after as StateValue,
        type: node.type as "initialization" | "instant",
        source: sourceMatch.source,
        description: node.description as string,
      })
    }
  })

  if (issues.length > 0) throw validationError(issues, input)
  return { nodes }
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

function isLengthFinishReason(reason: string | null): boolean {
  return reason !== null &&
    ["length", "max_tokens", "max_output_tokens"].includes(reason.toLowerCase())
}

class RateGate {
  readonly intervalMs: number
  nextStart = 0
  tail: Promise<void> = Promise.resolve()

  constructor(requestsPerSecond: number) {
    this.intervalMs = 1000 / requestsPerSecond
  }

  async wait(): Promise<void> {
    let release = (): void => {}
    const turn = new Promise<void>((resolveTurn) => { release = resolveTurn })
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
  job: StateJob,
  prompt: PromptParts,
  prenodes: StateNode[],
  retries: number,
  retryDelayMs: number,
  streamed: boolean,
  provider: ModelType,
  baseRuntimeParameters: JsonObject,
  debugLogger?: DebugLogger,
): Promise<StateResult> {
  const totalAttempts = retries + 1
  let reasoningEffort: "max" | "high" = "max"
  let retryFeedback: string | undefined
  let retryOutput: string | undefined
  let repairContext:
    | { result: JsonObject; nodeNames: string[] }
    | undefined

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const attemptNumber = attempt + 1
    const messages = buildMessages(
      prompt,
      job.text,
      job.characters,
      prenodes,
      retryFeedback,
      retryOutput,
    )
    const runtimeParameters: JsonObject = {
      ...baseRuntimeParameters,
      ...(provider === "DeepSeek"
        ? { reasoning_effort: reasoningEffort }
        : {}),
    }
    await limiter.wait()
    const startedAt = new Date()
    debugLogger?.start(
      attemptNumber,
      totalAttempts,
      startedAt.toISOString(),
      messages,
      { stream: streamed, ...runtimeParameters },
    )

    let received = false
    let thinking = ""
    let content = ""
    let output = extractLLMOutputInfo(null, null)
    const finish = (
      status: AttemptStatus,
      error?: unknown,
      parsedJson?: JsonObject,
      validatedResult?: StateResult,
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
        const stream = await llm.stream(messages, runtimeParameters as any)
        received = true
        const collected = await collectAssistantStream(
          stream,
          debugLogger ? (kind, chunk) => debugLogger.stream(kind, chunk) : undefined,
        )
        thinking = collected.thinking
        content = collected.content
        output = collected.output
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
      const length = isLengthFinishReason(output.finish_reason)
      finish(length ? "length" : received ? "response_error" : "request_error", error)
      if ((length || isTimeoutError(error)) && attempt < retries) {
        if (length && provider === "DeepSeek") reasoningEffort = "high"
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }

    if (isLengthFinishReason(output.finish_reason)) {
      const error = new Error(`LLM stopped because finish_reason=${output.finish_reason}.`)
      finish("length", error)
      if (attempt < retries) {
        if (provider === "DeepSeek") reasoningEffort = "high"
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }

    let parsedJson: JsonObject | undefined
    try {
      parsedJson = parseStateOutput(content)
      if (repairContext !== undefined) {
        parsedJson = mergeNodeRepairs(
          repairContext.result,
          parsedJson,
          repairContext.nodeNames,
        )
      }
    } catch (error) {
      const retryError = new RetryableResultError(
        invalidJsonRetryFeedback(content),
      )
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
      const result = validateResult(parsedJson, job.text)
      finish("success", undefined, parsedJson, result)
      return result
    } catch (error) {
      finish("validation_error", error, parsedJson)
      if (error instanceof RetryableResultError && attempt < retries) {
        retryFeedback = error.feedback
        retryOutput = JSON.stringify(parsedJson, null, 2)
        repairContext = {
          result: parsedJson,
          nodeNames: error.nodeNames,
        }
        await delay(retryDelayMs * attemptNumber)
        continue
      }
      throw error
    }
  }
  throw new Error("State extraction retry loop exhausted.")
}

function completedResults(
  path: string,
  jobs: StateJob[],
): Map<string, StateResult> {
  if (!existsSync(path)) return new Map()
  const jobByKey = new Map(
    jobs.map((job) => [jobKey(job.plotIndex, job.chunkIndex), job]),
  )
  const results = new Map<string, StateResult>()
  for (const line of readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean)) {
    const row = JSON.parse(line)
    if (!Number.isInteger(row.plot_index) || !Number.isInteger(row.chunk_index)) {
      throw new Error("Existing state JSONL contains an invalid job index.")
    }
    const key = jobKey(row.plot_index, row.chunk_index)
    const job = jobByKey.get(key)
    if (!job) throw new Error(`Existing state JSONL contains unknown job ${key}.`)
    if (results.has(key)) throw new Error(`Existing state JSONL duplicates job ${key}.`)
    if (!row.result || typeof row.result !== "object" || Array.isArray(row.result)) {
      throw new Error(`Existing state JSONL has no result for job ${key}.`)
    }
    results.set(key, validateResult(row.result, job.text))
  }
  return results
}

export function outputRecord(
  title: string,
  job: StateJob,
  result: StateResult,
  model: string,
): JsonObject {
  return {
    model,
    source_title: title,
    plot_index: job.plotIndex,
    chunk_index: job.chunkIndex,
    volume: job.volume,
    chapter: job.chapter,
    result,
  }
}

export function selectJobs(
  jobs: StateJob[],
  completed: Set<string>,
  start: number,
  limit?: number,
): StateJob[] {
  const pending = jobs.filter(
    (job) =>
      job.plotIndex >= start &&
      !completed.has(jobKey(job.plotIndex, job.chunkIndex)),
  )
  return limit === undefined ? pending : pending.slice(0, limit)
}

export function groupJobsByPlot(jobs: StateJob[]): StateJob[][] {
  const groups = new Map<number, StateJob[]>()
  for (const job of jobs) {
    const group = groups.get(job.plotIndex) ?? []
    group.push(job)
    groups.set(job.plotIndex, group)
  }
  return [...groups.values()]
    .map((group) => group.sort((a, b) => a.chunkIndex - b.chunkIndex))
    .sort((a, b) => a[0].plotIndex - b[0].plotIndex)
}

function isPresetState(node: StateNode): boolean {
  return node.name.endsWith(".active") || node.name.endsWith(".location")
}

export function previousNodes(
  job: StateJob,
  results: Map<string, StateResult>,
): StateNode[] {
  if (job.chunkIndex === 0) return []

  const earlierResults: StateResult[] = []
  for (let chunkIndex = 0; chunkIndex < job.chunkIndex; chunkIndex += 1) {
    const result = results.get(jobKey(job.plotIndex, chunkIndex))
    if (!result) {
      throw new Error(
        `${job.jobId} requires the result of plot ${job.plotIndex}, ` +
          `chunk ${chunkIndex}.`,
      )
    }
    earlierResults.push(result)
  }

  const previous = earlierResults.at(-1)!
  const previousPresetNames = new Set(
    previous.nodes.filter(isPresetState).map((node) => node.name),
  )
  const trackedPresets = new Map<string, StateNode>()
  for (const result of earlierResults) {
    for (const node of result.nodes) {
      if (isPresetState(node)) trackedPresets.set(node.name, node)
    }
  }
  for (const name of previousPresetNames) trackedPresets.delete(name)

  return [...trackedPresets.values(), ...previous.nodes]
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
    process.env.RLFF_STATE_MODEL ??
    process.env.RLFF_PLOT_MODEL ??
    process.env.RLFF_VERIFIER_MODEL
  const args: Omit<Args, "inputJson" | "outputJsonl"> = {
    prompt: resolve(projectRoot(), "state_extraction.txt"),
    provider: providerValue(
      process.env.RLFF_STATE_PROVIDER ??
        process.env.RLFF_PLOT_PROVIDER ??
        "DeepSeek",
    ),
    model: configuredModel ?? "",
    baseURL: undefined,
    concurrency: 1,
    requestsPerSecond: 5,
    retries: 2,
    retryDelayMs: 2000,
    timeoutMs: 120_000,
    maxTokens: 8 * 1024,
    temperature: 0,
    start: 0,
    limit: undefined,
    overwrite: false,
    dryRun: false,
    debug: false,
    stream: false,
  }
  const numericOptions: Record<string, keyof typeof args> = {
    "--concurrency": "concurrency",
    "--requests-per-second": "requestsPerSecond",
    "--retries": "retries",
    "--retry-delay-ms": "retryDelayMs",
    "--timeout-ms": "timeoutMs",
    "--max-tokens": "maxTokens",
    "--temperature": "temperature",
    "--start": "start",
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
        ;(args as any)[numericOptions[option]] = numericValue(value, option)
      } else throw new Error(`Unknown option: ${option}`)
      index += 1
    }
  }

  if (positional.length !== 2) {
    throw new Error("Usage: extract_states.ts <input_json> <output_jsonl> [options]")
  }
  for (const [name, value, minimum] of [
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
  if (args.temperature < 0 || args.temperature > 2) {
    throw new Error("temperature must be between 0 and 2.")
  }
  if (!Number.isInteger(args.start) || args.start < 0) {
    throw new Error("start must be a non-negative integer.")
  }
  if (args.limit !== undefined && (!Number.isInteger(args.limit) || args.limit < 1)) {
    throw new Error("limit must be a positive integer.")
  }
  if (!args.model) {
    if (args.provider === "DeepSeek") args.model = "deepseek-v4-pro"
    else throw new Error("--model or RLFF_STATE_MODEL is required for GLM and Qwen.")
  }
  args.baseURL ??= {
    DeepSeek: process.env.DEEPSEEK_BASE_URL,
    GLM: process.env.GLM_BASE_URL,
    Qwen: process.env.DASHSCOPE_BASE_URL,
  }[args.provider]
  return { ...args, inputJson: positional[0], outputJsonl: positional[1] }
}

async function run(args: Args): Promise<void> {
  const loaded = loadJobs(args.inputJson)
  const prompt = splitPrompt(
    readFileSync(args.prompt, "utf8").replace(/^\uFEFF/, ""),
  )
  const results = args.overwrite
    ? new Map<string, StateResult>()
    : completedResults(args.outputJsonl, loaded.allJobs)
  const jobs = selectJobs(
    loaded.jobs,
    new Set(results.keys()),
    args.start,
    args.limit,
  )
  const plotGroups = groupJobsByPlot(jobs)

  if (args.overwrite && !args.dryRun && existsSync(args.outputJsonl)) {
    unlinkSync(args.outputJsonl)
  }
  console.log(JSON.stringify({
    provider: args.provider,
    model: args.model,
    source_title: loaded.title,
    plot_count: loaded.plotCount,
    skipped_short_plot_count: loaded.skippedPlotCount,
    minimum_plot_tokens: MIN_PLOT_TOKENS,
    source_chunk_count: loaded.allJobs.length,
    chunk_count: loaded.jobs.length,
    selected_count: jobs.length,
    selected_plot_count: plotGroups.length,
    existing_count: results.size,
    start: args.start,
    concurrency: args.concurrency,
    debug: args.debug,
    stream: args.stream,
  }, null, 2))

  if (args.dryRun) {
    if (jobs.length > 0) {
      console.log(JSON.stringify(
        buildMessages(
          prompt,
          jobs[0].text,
          jobs[0].characters,
          previousNodes(jobs[0], results),
        ),
        null,
        2,
      ))
    }
    return
  }
  if (jobs.length === 0) return

  const providerParameters: JsonObject =
    args.provider === "DeepSeek"
      ? { reasoning_effort: "max" }
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
    retries: args.retries,
    retry_delay_ms: args.retryDelayMs,
    requests_per_second: args.requestsPerSecond,
    concurrency: args.concurrency,
    stream: args.stream,
    ...(args.baseURL ? { base_url: args.baseURL } : {}),
    ...providerParameters,
  }

  let groupCursor = 0
  let completed = 0
  let blocked = 0
  const failures: Array<{ job_id: string; error: string }> = []

  async function worker(): Promise<void> {
    while (true) {
      const plotGroup = plotGroups[groupCursor++]
      if (!plotGroup) return
      for (let index = 0; index < plotGroup.length; index += 1) {
        const job = plotGroup[index]
        try {
          const prenodes = previousNodes(job, results)
          const logPath = args.debug
            ? debugLogPath(args.inputJson, job.jobId)
            : undefined
          if (logPath) {
            initializeDebugLog(logPath, loaded.title, job, debugCallParameters)
          }
          const result = await extractOne(
            llm,
            limiter,
            job,
            prompt,
            prenodes,
            args.retries,
            args.retryDelayMs,
            args.stream,
            args.provider,
            clientConfig,
            logPath ? createDebugLogger(logPath, args.stream) : undefined,
          )
          results.set(jobKey(job.plotIndex, job.chunkIndex), result)
          appendFileSync(
            args.outputJsonl,
            JSON.stringify(outputRecord(loaded.title, job, result, args.model)) + "\n",
            "utf8",
          )
          completed += 1
          console.log(`[${completed}/${jobs.length}] ${job.jobId}`)
        } catch (error) {
          failures.push({
            job_id: job.jobId,
            error: error instanceof Error ? error.message : String(error),
          })
          blocked += plotGroup.length - index - 1
          break
        }
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(args.concurrency, plotGroups.length) }, worker),
  )
  console.log(JSON.stringify({
    completed,
    failed: failures.length,
    blocked,
    failures,
  }, null, 2))
  if (failures.length > 0) {
    throw new Error(`${failures.length} state extraction job(s) failed.`)
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
