import {
  Archive,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  FileText,
  FolderOpen,
  LoaderCircle,
  MessageSquareText,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
import {
  FormEvent,
  MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";

type Project = {
  name: string;
  path: string;
  model_ready: boolean;
  vision_model_ready?: boolean;
  ocr?: {
    available: boolean;
    cuda: boolean;
    model_dir: string;
    reason?: string | null;
  };
};
type RunStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "cancelled";
type ChatSummary = {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  archived_at?: string | null;
  run_id?: string | null;
  run_status?: RunStatus | null;
  extraction_task?: string | null;
  extraction_status?: string | null;
};
type ChatMessage = {
  id: number;
  role: string;
  content: string;
  kind: "message" | "event";
  created_at: string;
  tool_calls: unknown[];
};
type MessagePage = { items: ChatMessage[]; next_before?: number | null };
type ApprovalTool = {
  name: string;
  description: string;
  arguments: Record<string, unknown>;
  effect: string;
  risk: string;
  reason: string;
};
type ApprovalRequest = {
  token: string;
  expires_in_seconds: number;
  tools: ApprovalTool[];
};
type ChatRun = {
  chat_session: number;
  run_id: string;
  status: RunStatus;
  content?: string;
  tool_calls?: number;
  error?: string | null;
  approval?: ApprovalRequest | null;
};
type ExtractionState = {
  task: string;
  chat_session: number;
  status: string;
  stage?: string;
  dataset_root?: string;
  error?: string;
};

const terminalRuns = new Set<RunStatus>(["completed", "failed", "cancelled"]);
const terminalExtractions = new Set(["completed", "failed", "incomplete"]);
const stageNames: Record<string, string> = {
  chunk: "整理原文",
  plot_extraction: "提取剧情",
  character_profile: "整理角色并生成档案",
  plot_reconstruction: "重建剧情",
  dialogue_extraction: "提取对话",
  dataset: "生成数据集",
  image_scan: "扫描漫画页面",
  visual_extraction: "提取漫画内容",
  ocr_alignment: "校正对白文字",
  character_catalog: "建立角色名称参考",
  character_assignment: "映射局部角色",
  chapter_reconstruction: "按话重建剧情",
  dialogue_revision: "修订说话者与对白",
  export: "导出训练数据",
};
const stageDescriptions: Record<string, string> = {
  chunk: "按分卷和章节规则切分原文",
  plot_extraction: "从文本片段识别剧情与角色",
  character_profile: "合并角色名称与描述并生成档案",
  plot_reconstruction: "按照剧情边界重建连续文本",
  dialogue_extraction: "从重建后的剧情中提取对话",
  dataset: "整理最终结果并导出 ShareGPT",
  image_scan: "验证、排序图片并建立五页批次",
  visual_extraction: "使用视觉模型并行提取对白、角色特征与剧情",
  ocr_alignment: "使用可选 OCR 结果修正 VLM 对白",
  character_catalog: "根据全部角色观察生成正式名称参考表",
  character_assignment: "将各批次局部角色映射到正式名称",
  chapter_reconstruction: "根据页面中的话标题重建剧情与对白",
  dialogue_revision: "以文字模型再次确认每话说话者与对白",
  export: "生成 ShareGPT 训练样本",
};

const headingSplitterSeparator = /[,，/、\\;；\s]+/;
const headingNumber = /第[零〇一二两三四五六七八九十百千万兩\d]+/;

function parseHeadingSplitters(value: string): string[] {
  const splitters: string[] = [];
  for (const item of value.trim().split(headingSplitterSeparator)) {
    if (!item) continue;
    const normalized = item.replace(headingNumber, "第{num}");
    if (!splitters.includes(normalized)) splitters.push(normalized);
  }
  return splitters;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    const error = new Error(detail.detail ?? "请求失败");
    Object.assign(error, { status: response.status });
    throw error;
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

function errorStatus(error: unknown): number | undefined {
  return error instanceof Error
    ? (error as Error & { status?: number }).status
    : undefined;
}

function displayTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Workspace />} />
      <Route path="/chats/new" element={<Workspace />} />
      <Route path="/chats/:chatId" element={<Workspace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function Workspace() {
  const { chatId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const selectedId =
    chatId && /^\d+$/.test(chatId) ? Number.parseInt(chatId, 10) : undefined;
  const isNew = location.pathname === "/chats/new";
  const [project, setProject] = useState<Project>();
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [archived, setArchived] = useState<ChatSummary[]>([]);
  const [archivesOpen, setArchivesOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [nextBefore, setNextBefore] = useState<number>();
  const [run, setRun] = useState<ChatRun>();
  const [localRuns, setLocalRuns] = useState<Record<number, string>>({});
  const [unread, setUnread] = useState<Set<number>>(new Set());
  const [draft, setDraft] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [showExtraction, setShowExtraction] = useState(false);
  const [extraction, setExtraction] = useState<ExtractionState>();
  const [error, setError] = useState<string>();
  const bottomRef = useRef<HTMLDivElement>(null);
  const previousRuns = useRef<Map<number, string>>(new Map());

  const selectedChat = useMemo(
    () => [...chats, ...archived].find((item) => item.id === selectedId),
    [archived, chats, selectedId],
  );
  const selectedRunId =
    (selectedId ? localRuns[selectedId] : undefined) ??
    selectedChat?.run_id ??
    undefined;
  const busy = Boolean(
    selectedRunId &&
      (!run || run.run_id !== selectedRunId || !terminalRuns.has(run.status)),
  );
  const localExtractionStatus =
    extraction &&
    extraction.chat_session === selectedId &&
    !terminalExtractions.has(extraction.status)
      ? extraction.status
      : undefined;
  const selectedExtractionStatus =
    localExtractionStatus ?? selectedChat?.extraction_status ?? undefined;
  const extractionActive = Boolean(selectedExtractionStatus);
  const workflowBlocksInput = Boolean(
    selectedExtractionStatus &&
      selectedExtractionStatus !== "awaiting_confirmation",
  );
  const readOnly = Boolean(selectedChat?.archived_at);

  const applyActiveChats = useCallback((activeValues: ChatSummary[]) => {
    const currentRuns = new Map<number, string>();
    for (const chat of activeValues) {
      const activity = chat.run_status ?? chat.extraction_status;
      if (activity) currentRuns.set(chat.id, activity);
    }
    setUnread((current) => {
      const next = new Set(current);
      for (const [id] of previousRuns.current) {
        if (!currentRuns.has(id) && id !== selectedId) next.add(id);
      }
      return next;
    });
    previousRuns.current = currentRuns;
    setChats(activeValues);
  }, [selectedId]);

  const refreshActiveChats = useCallback(async () => {
    applyActiveChats(await api<ChatSummary[]>("/api/chats"));
  }, [applyActiveChats]);

  const refreshArchivedChats = useCallback(async () => {
    setArchived(await api<ChatSummary[]>("/api/chats?archived=true"));
  }, []);

  const refreshChats = useCallback(async () => {
    await Promise.all([refreshActiveChats(), refreshArchivedChats()]);
  }, [refreshActiveChats, refreshArchivedChats]);

  useEffect(() => {
    void Promise.all([
      api<Project>("/api/project").then(setProject),
      refreshChats(),
    ]).catch((requestError) =>
      setError(
        requestError instanceof Error
          ? requestError.message
          : "无法连接本地服务",
      ),
    );
  }, [refreshChats]);

  useEffect(() => {
    const hasActiveTasks = chats.some(
      (chat) => chat.run_status || chat.extraction_status,
    );
    const timer = window.setInterval(
      () => void refreshActiveChats(),
      hasActiveTasks ? 1500 : 10_000,
    );
    return () => window.clearInterval(timer);
  }, [chats, refreshActiveChats]);

  useEffect(() => {
    if (location.pathname !== "/" || !project) return;
    const saved = window.localStorage.getItem("rar:last-chat:" + project.path);
    const savedId = saved ? Number.parseInt(saved, 10) : undefined;
    const all = [...chats, ...archived];
    const target =
      savedId && all.some((chat) => chat.id === savedId)
        ? "/chats/" + savedId
        : chats[0]
          ? "/chats/" + chats[0].id
          : "/chats/new";
    navigate(target, { replace: true });
  }, [archived, chats, location.pathname, navigate, project]);

  const loadMessages = useCallback(
    async (id: number) => {
      try {
        const page = await api<MessagePage>(
          "/api/chats/" + id + "/messages?limit=100",
        );
        setMessages(page.items);
        setNextBefore(page.next_before ?? undefined);
        setUnread((current) => {
          const next = new Set(current);
          next.delete(id);
          return next;
        });
      } catch (requestError) {
        if (errorStatus(requestError) === 404) {
          setError("该对话不存在或已经被删除。");
          navigate("/", { replace: true });
        } else {
          setError(
            requestError instanceof Error ? requestError.message : "无法读取对话",
          );
        }
      }
    },
    [navigate],
  );

  useEffect(() => {
    setRun(undefined);
    if (!selectedId) {
      setMessages([]);
      setNextBefore(undefined);
      return;
    }
    if (!project) return;
    window.localStorage.setItem(
      "rar:last-chat:" + project.path,
      String(selectedId),
    );
    void loadMessages(selectedId);
  }, [loadMessages, project, selectedId]);

  useEffect(() => {
    const key = project
      ? "rar:draft:" + project.path + ":" + (selectedId ?? "new")
      : undefined;
    setDraft(key ? window.localStorage.getItem(key) ?? "" : "");
  }, [project, selectedId]);

  useEffect(() => {
    if (!project) return;
    window.localStorage.setItem(
      "rar:draft:" + project.path + ":" + (selectedId ?? "new"),
      draft,
    );
  }, [draft, project, selectedId]);

  useEffect(() => {
    if (!selectedId || !selectedRunId) {
      setRun(undefined);
      return;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const value = await api<ChatRun>(
          "/api/chats/" + selectedId + "/runs/" + selectedRunId,
        );
        if (stopped) return;
        setRun(value);
        if (terminalRuns.has(value.status)) {
          setLocalRuns((current) => {
            const next = { ...current };
            delete next[selectedId];
            return next;
          });
          await Promise.all([loadMessages(selectedId), refreshChats()]);
          return;
        }
        window.setTimeout(() => void poll(), 750);
      } catch (requestError) {
        if (!stopped && errorStatus(requestError) !== 404) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "无法读取运行状态",
          );
        }
      }
    };
    void poll();
    return () => {
      stopped = true;
    };
  }, [loadMessages, refreshChats, selectedId, selectedRunId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, run?.approval]);

  useEffect(() => {
    const task = selectedChat?.extraction_task;
    if (!selectedId || !task) {
      if (extraction?.chat_session !== selectedId) setExtraction(undefined);
      return;
    }
    if (extraction?.task === task) return;
    void api<ExtractionState>("/api/extractions/" + task)
      .then(setExtraction)
      .catch((requestError) => {
        if (errorStatus(requestError) !== 404) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "无法读取提取任务状态",
          );
        }
      });
  }, [extraction?.chat_session, extraction?.task, selectedChat, selectedId]);

  useEffect(() => {
    if (
      !extraction ||
      ["completed", "failed", "incomplete"].includes(extraction.status)
    ) {
      return;
    }
    const timer = window.setInterval(async () => {
      const value = await api<ExtractionState>(
        "/api/extractions/" + extraction.task,
      );
      setExtraction(value);
      if (["completed", "failed", "incomplete"].includes(value.status)) {
        window.clearInterval(timer);
        if (value.chat_session === selectedId) {
          void loadMessages(value.chat_session);
        }
        void refreshChats();
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [extraction, loadMessages, refreshChats, selectedId]);

  async function sendMessage() {
    const value = draft.trim();
    if (!value || busy || workflowBlocksInput || readOnly) return;
    setDraft("");
    setError(undefined);
    try {
      const accepted = await api<ChatRun>("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: value, chat_session: selectedId }),
      });
      setLocalRuns((current) => ({
        ...current,
        [accepted.chat_session]: accepted.run_id,
      }));
      if (!selectedId) {
        navigate("/chats/" + accepted.chat_session, { replace: true });
      } else {
        await loadMessages(selectedId);
      }
      await refreshChats();
    } catch (requestError) {
      setDraft(value);
      setError(
        requestError instanceof Error ? requestError.message : "消息发送失败",
      );
    }
  }

  async function loadOlder() {
    if (!selectedId || !nextBefore) return;
    const page = await api<MessagePage>(
      "/api/chats/" +
        selectedId +
        "/messages?limit=100&before=" +
        nextBefore,
    );
    setMessages((current) => [...page.items, ...current]);
    setNextBefore(page.next_before ?? undefined);
  }

  async function resolveApproval(allowed: boolean) {
    if (!run?.approval) return;
    const accepted = await api<ChatRun>(
      "/api/approvals/" + run.approval.token,
      { method: "POST", body: JSON.stringify({ allowed }) },
    );
    setRun({ ...accepted, approval: null });
    setLocalRuns((current) => ({
      ...current,
      [accepted.chat_session]: accepted.run_id,
    }));
  }

  async function cancelRun() {
    if (!selectedId || !selectedRunId) return;
    const value = await api<ChatRun>(
      "/api/chats/" +
        selectedId +
        "/runs/" +
        selectedRunId +
        "/cancel",
      { method: "POST" },
    );
    setRun(value);
    await Promise.all([loadMessages(selectedId), refreshChats()]);
  }

  async function renameChat(chat: ChatSummary) {
    const title = window.prompt("输入新的对话标题", chat.title)?.trim();
    if (!title) return;
    await api("/api/chats/" + chat.id, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    await refreshChats();
  }

  async function archiveChat(chat: ChatSummary, value: boolean) {
    await api("/api/chats/" + chat.id, {
      method: "PATCH",
      body: JSON.stringify({ archived: value }),
    });
    await refreshChats();
    if (value && chat.id === selectedId) {
      const next = chats.find((item) => item.id !== chat.id);
      navigate(next ? "/chats/" + next.id : "/chats/new");
    }
  }

  async function deleteChat(chat: ChatSummary) {
    if (
      !window.confirm(
        "永久删除对话“" +
          chat.title +
          "”？项目文件和数据集不会被删除。",
      )
    ) {
      return;
    }
    await api("/api/chats/" + chat.id, { method: "DELETE" });
    await refreshChats();
    if (chat.id === selectedId) {
      const next = chats.find((item) => item.id !== chat.id);
      navigate(next ? "/chats/" + next.id : "/chats/new");
    }
  }

  async function confirmStage() {
    if (!extraction) return;
    await api("/api/extractions/" + extraction.task + "/confirm", {
      method: "POST",
    });
    setExtraction((value) =>
      value ? { ...value, status: "running" } : value,
    );
  }

  const visibleMessages = messages.filter(
    (message) =>
      message.kind === "event" ||
      (message.role !== "tool" &&
        (message.content.trim() || message.role === "user")),
  );

  return (
    <div className="app-shell">
      <aside className={"sidebar " + (sidebarOpen ? "open" : "closed")}>
        <div className="brand-row">
          <div className="brand-mark">R</div>
          <div className="brand-copy">
            <strong>RAR</strong>
            <span>Read And Retrieve</span>
          </div>
          <button
            className="icon-button mobile-only"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>
        <button className="new-chat" onClick={() => navigate("/chats/new")}>
          <Plus size={17} /> 新对话
        </button>
        <div className="side-section">
          <div className="side-title">
            <FolderOpen size={14} /> 当前项目
          </div>
          <div className="project-tree-root">
            <span className="project-dot" />
            <div>
              <strong>{project?.name ?? "正在连接..."}</strong>
              <small>{project?.path}</small>
            </div>
          </div>
        </div>
        <div className="chat-tree">
          <div className="side-title">
            <MessageSquareText size={14} /> 对话 <span>{chats.length}</span>
          </div>
          <div className="chat-list">
            {chats.map((chat) => (
              <ChatItem
                key={chat.id}
                chat={chat}
                active={chat.id === selectedId}
                unread={unread.has(chat.id)}
                onOpen={() => navigate("/chats/" + chat.id)}
                onRename={() => void renameChat(chat)}
                onArchive={() => void archiveChat(chat, true)}
                onDelete={() => void deleteChat(chat)}
              />
            ))}
            {chats.length === 0 && (
              <p className="empty-side">项目中的对话会显示在这里。</p>
            )}
          </div>
          {archived.length > 0 && (
            <div className="archive-section">
              <button
                className="archive-toggle"
                onClick={() => setArchivesOpen((value) => !value)}
              >
                {archivesOpen ? (
                  <ChevronDown size={13} />
                ) : (
                  <ChevronRight size={13} />
                )}
                已归档 <span>{archived.length}</span>
              </button>
              {archivesOpen &&
                archived.map((chat) => (
                  <ChatItem
                    key={chat.id}
                    chat={chat}
                    active={chat.id === selectedId}
                    unread={false}
                    onOpen={() => navigate("/chats/" + chat.id)}
                    onRename={() => void renameChat(chat)}
                    onArchive={() => void archiveChat(chat, false)}
                    onDelete={() => void deleteChat(chat)}
                  />
                ))}
            </div>
          )}
        </div>
        <div className="sidebar-footer">
          <span
            className={
              "status-light " + (project?.model_ready ? "ready" : "")
            }
          />
          {project?.model_ready ? "模型已连接" : "等待模型配置"}
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen((value) => !value)}
          >
            {sidebarOpen ? (
              <PanelLeftClose size={19} />
            ) : (
              <PanelLeftOpen size={19} />
            )}
          </button>
          <div className="topbar-title">
            <span className="topbar-dot" />
            {selectedChat?.title ?? "项目助手"}
          </div>
          {readOnly && (
            <button
              className="restore-button"
              onClick={() => void archiveChat(selectedChat!, false)}
            >
              <RotateCcw size={14} /> 恢复对话
            </button>
          )}
          <button
            className="extract-button"
            disabled={readOnly || busy || extractionActive}
            onClick={() => setShowExtraction(true)}
          >
            <Sparkles size={16} /> 提取数据集
          </button>
        </header>

        <section
          className={
            "conversation " +
            (isNew && visibleMessages.length === 0 ? "welcome-state" : "")
          }
        >
          {isNew && visibleMessages.length === 0 ? (
            <Welcome
              onExtract={() => setShowExtraction(true)}
              onInspect={() =>
                setDraft("检查当前项目中已有的数据集和中间产物。")
              }
            />
          ) : (
            <div className="message-stream">
              {nextBefore && (
                <button className="load-older" onClick={() => void loadOlder()}>
                  加载更早的消息
                </button>
              )}
              {visibleMessages.map((message) =>
                message.kind === "event" ? (
                  <div className="timeline-event" key={message.id}>
                    <span />
                    <p>{message.content}</p>
                  </div>
                ) : (
                  <article
                    className={"message " + message.role}
                    key={message.id}
                  >
                    <div className="message-avatar">
                      {message.role === "assistant" ? "R" : "你"}
                    </div>
                    <div>
                      <span>
                        {message.role === "assistant" ? "RAR" : "你"}
                      </span>
                      <p>{message.content || "（无文本输出）"}</p>
                    </div>
                  </article>
                ),
              )}
              {run?.status === "awaiting_approval" && run.approval && (
                <ApprovalCard
                  approval={run.approval}
                  onResolve={(value) => void resolveApproval(value)}
                />
              )}
              {busy && run?.status !== "awaiting_approval" && (
                <article className="message assistant">
                  <div className="message-avatar">R</div>
                  <div>
                    <span>RAR</span>
                    <p className="thinking">
                      <i />
                      <i />
                      <i />
                    </p>
                  </div>
                </article>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        {extraction &&
          !["completed", "failed", "incomplete"].includes(
            extraction.status,
          ) && (
            <div className={"run-banner " + extraction.status}>
              {extraction.status === "awaiting_confirmation" ? (
                <Check size={17} />
              ) : (
                <LoaderCircle className="spin" size={17} />
              )}
              <span>
                <strong>
                  {extraction.status === "awaiting_confirmation"
                    ? `${stageNames[extraction.stage ?? ""] ?? "当前阶段"}已完成`
                    : `正在${stageNames[extraction.stage ?? ""] ?? "准备资源"}`}
                </strong>
                {stageDescriptions[extraction.stage ?? ""] ??
                  "正在初始化提取任务"}
              </span>
              {extraction.status === "awaiting_confirmation" && (
                <button
                  className="confirm-button"
                  disabled={busy}
                  title={busy ? "请等待当前对话任务完成" : undefined}
                  onClick={() => void confirmStage()}
                >
                  确认并继续
                </button>
              )}
            </div>
          )}
        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={() => setError(undefined)}>
              <X size={15} />
            </button>
          </div>
        )}

        <div className="composer-wrap">
          <form
            className={"composer " + (workflowBlocksInput ? "locked" : "")}
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
          >
            <textarea
              disabled={busy || workflowBlocksInput || readOnly}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder={
                readOnly
                  ? "该对话已归档，请恢复后继续"
                  : busy
                    ? "当前对话正在执行任务"
                    : workflowBlocksInput
                      ? "数据集提取正在运行，阶段完成后可继续对话"
                    : "告诉 RAR 你想提取、检查或修改什么..."
              }
              rows={1}
            />
            {busy ? (
              <button
                type="button"
                className="stop-button"
                title="停止任务"
                onClick={() => void cancelRun()}
              >
                <Square size={15} fill="currentColor" />
              </button>
            ) : (
              <button
                className="send-button"
                disabled={!draft.trim() || workflowBlocksInput || readOnly}
              >
                <ArrowUp size={18} />
              </button>
            )}
          </form>
          <p className="composer-hint">
            Enter 发送 · Shift + Enter 换行 · 所有结果仅保存在本地
          </p>
        </div>
      </main>

      {showExtraction && (
        <ExtractionDialog
          chatSession={selectedId}
          project={project}
          onClose={() => setShowExtraction(false)}
          onStarted={(value) => {
            setExtraction(value);
            setShowExtraction(false);
            navigate("/chats/" + value.chat_session, {
              replace: !selectedId,
            });
            void refreshChats();
          }}
        />
      )}
    </div>
  );
}

function Welcome({
  onExtract,
  onInspect,
}: {
  onExtract: () => void;
  onInspect: () => void;
}) {
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <MessageSquareText size={27} />
      </div>
      <p className="eyebrow">LOCAL DATASET STUDIO</p>
      <h1>
        把故事，整理成
        <br />
        <em>可追溯的对话。</em>
      </h1>
      <p className="welcome-copy">
        发送第一条消息后才会创建对话。你也可以直接启动文本提取，
        RAR 会自动创建用于展示进度的对话窗口。
      </p>
      <div className="suggestions">
        <button onClick={onExtract}>
          <FileText size={18} />
          <span>
            <strong>从文本开始提取</strong>
            <small>创建剧情、角色与对话数据</small>
          </span>
          <ChevronRight size={15} />
        </button>
        <button onClick={onInspect}>
          <FolderOpen size={18} />
          <span>
            <strong>检查当前项目</strong>
            <small>了解资源与已有结果</small>
          </span>
          <ChevronRight size={15} />
        </button>
      </div>
    </div>
  );
}

function ChatItem({
  chat,
  active,
  unread,
  onOpen,
  onRename,
  onArchive,
  onDelete,
}: {
  chat: ChatSummary;
  active: boolean;
  unread: boolean;
  onOpen: () => void;
  onRename: () => void;
  onArchive: () => void;
  onDelete: () => void;
}) {
  const [menuPosition, setMenuPosition] = useState<{
    left: number;
    top: number;
  }>();
  const busy = Boolean(chat.run_status || chat.extraction_status);

  function toggleMenu(event: ReactMouseEvent<HTMLButtonElement>) {
    if (menuPosition) {
      setMenuPosition(undefined);
      return;
    }
    const trigger = event.currentTarget.getBoundingClientRect();
    const menuWidth = 125;
    const menuHeight = 96;
    const edgeGap = 8;
    const itemGap = 4;
    const top =
      trigger.bottom + itemGap + menuHeight <= window.innerHeight - edgeGap
        ? trigger.bottom + itemGap
        : Math.max(edgeGap, trigger.top - menuHeight - itemGap);
    const left = Math.min(
      window.innerWidth - menuWidth - edgeGap,
      Math.max(edgeGap, trigger.right - menuWidth),
    );
    setMenuPosition({ left, top });
  }

  return (
    <div className={"chat-tree-item " + (active ? "active" : "")}>
      <button className="chat-open" onClick={onOpen}>
        <span
          className={
            "chat-state " +
            (chat.run_status ?? chat.extraction_status ?? "") +
            " " +
            (unread ? "unread" : "")
          }
        />
        <span>
          <strong>{chat.title}</strong>
          <small>
            {chat.run_status === "awaiting_approval"
              ? "等待授权"
              : chat.run_status
                ? "正在运行"
                : chat.extraction_status
                  ? "正在提取数据集"
                : displayTime(chat.updated_at)}
          </small>
        </span>
      </button>
      <button
        className="chat-menu-button"
        aria-label={`打开“${chat.title}”的菜单`}
        aria-expanded={Boolean(menuPosition)}
        onClick={toggleMenu}
      >
        <MoreHorizontal size={14} />
      </button>
      {menuPosition &&
        createPortal(
          <div
            className="chat-menu chat-menu-floating"
            role="menu"
            style={menuPosition}
            onMouseLeave={() => setMenuPosition(undefined)}
          >
            <button
              role="menuitem"
              onClick={() => {
                setMenuPosition(undefined);
                onRename();
              }}
            >
              重命名
            </button>
            <button
              role="menuitem"
              disabled={busy}
              onClick={() => {
                setMenuPosition(undefined);
                onArchive();
              }}
            >
              {chat.archived_at ? (
                <>
                  <RotateCcw size={12} /> 恢复
                </>
              ) : (
                <>
                  <Archive size={12} /> 归档
                </>
              )}
            </button>
            <button
              role="menuitem"
              className="danger"
              disabled={busy}
              onClick={() => {
                setMenuPosition(undefined);
                onDelete();
              }}
            >
              <Trash2 size={12} /> 永久删除
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}

function ApprovalCard({
  approval,
  onResolve,
}: {
  approval: ApprovalRequest;
  onResolve: (allowed: boolean) => void;
}) {
  return (
    <article className="approval-card">
      <div className="approval-heading">
        <span>
          <ShieldAlert size={17} />
        </span>
        <div>
          <strong>RAR 请求执行一项受保护操作</strong>
          <small>工具尚未执行，请确认后继续</small>
        </div>
      </div>
      <div className="approval-tools">
        {approval.tools.map((tool, index) => {
          const target =
            tool.arguments.path ?? tool.arguments.dataset_root;
          return (
            <div className="approval-tool" key={tool.name + "-" + index}>
              <div>
                <strong>{tool.name}</strong>
                <span>
                  {tool.effect} · {tool.risk}
                </span>
              </div>
              {typeof target === "string" && <code>{target}</code>}
              <p>{tool.reason || tool.description}</p>
            </div>
          );
        })}
      </div>
      <div className="approval-actions">
        <button onClick={() => onResolve(false)}>拒绝</button>
        <button className="allow" onClick={() => onResolve(true)}>
          允许一次
        </button>
      </div>
    </article>
  );
}

function ExtractionDialog({
  chatSession,
  project,
  onClose,
  onStarted,
}: {
  chatSession?: number;
  project?: Project;
  onClose: () => void;
  onStarted: (state: ExtractionState) => void;
}) {
  const [workflowType, setWorkflowType] = useState<"text" | "manga">("text");
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [volumeSplitters, setVolumeSplitters] = useState("");
  const [chapterSplitters, setChapterSplitters] = useState("");
  const [mode, setMode] = useState<"automatic" | "staged">("automatic");
  const [debug, setDebug] = useState(false);
  const [imageBatchSize, setImageBatchSize] = useState(5);
  const [ocrEnabled, setOcrEnabled] = useState(false);
  const [preview, setPreview] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(undefined);
    try {
      const volumes = parseHeadingSplitters(volumeSplitters);
      const chapters = parseHeadingSplitters(chapterSplitters);
      const value = await api<ExtractionState>("/api/extractions", {
        method: "POST",
        body: JSON.stringify({
          chat_session: chatSession,
          workflow_type: workflowType,
          mode,
          debug,
          ...(workflowType === "text" && volumes.length > 0
            ? { volume_splitters: volumes }
            : {}),
          ...(workflowType === "text" && chapters.length > 0
            ? { chapter_splitters: chapters }
            : {}),
          ...(workflowType === "manga"
            ? {
                vision_model: "qwen3.7-flash",
                image_batch_size: imageBatchSize,
                ocr_enabled: ocrEnabled,
              }
            : {}),
          manifest: {
            name,
            meta: {},
            resources: [
              {
                path,
                resource_type: workflowType,
                display_name: name,
                narrative_order: 0,
                meta: {},
              },
            ],
          },
        }),
      });
      onStarted(value);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "无法开始提取",
      );
    } finally {
      setBusy(false);
    }
  }

  async function previewImages() {
    setBusy(true);
    setError(undefined);
    setPreview(undefined);
    try {
      const query = new URLSearchParams({
        path,
        batch_size: String(imageBatchSize),
        debug: String(debug),
      });
      const value = await api<{
        image_count: number;
        batch_count: number;
        first_paths: string[];
      }>("/api/resources/manga/preview?" + query);
      setPreview(
        `识别到 ${value.image_count} 张图片、${value.batch_count} 个批次。` +
          (value.first_paths.length
            ? ` 排序开头：${value.first_paths.join("、")}`
            : ""),
      );
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "无法预览图片",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <form className="dialog" onSubmit={submit}>
        <div className="dialog-heading">
          <div className="dialog-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <p>NEW DATASET</p>
            <h2>{workflowType === "manga" ? "从漫画提取对话" : "从文本提取对话"}</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <p className="dialog-copy">
          {workflowType === "manga"
            ? "RAR 会扫描本地图像，以 VLM 提取对白与剧情，统一角色名称，按话重建并修订对话。"
            : "RAR 会依次整理原文、识别剧情、整理角色并生成档案、重建剧情、提取对白，并生成 ShareGPT 数据。"}
        </p>
        <fieldset className="mode-field">
          <legend>资源类型</legend>
          <button
            type="button"
            className={workflowType === "text" ? "selected" : ""}
            onClick={() => setWorkflowType("text")}
          >
            <strong>文字</strong>
            <small>TXT 等本地文本文件</small>
          </button>
          <button
            type="button"
            className={workflowType === "manga" ? "selected" : ""}
            onClick={() => setWorkflowType("manga")}
          >
            <strong>漫画</strong>
            <small>按页码排序的本地图像目录</small>
          </button>
        </fieldset>
        <label>
          数据集名称
          <input
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="例如：我的青春恋爱物语"
          />
        </label>
        <label>
          {workflowType === "manga" ? "漫画图像文件夹" : "文本文件路径"}
          <input
            required
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder={
              workflowType === "manga" ? "resources/manga" : "resources/book.txt"
            }
          />
          <small>填写当前项目内的相对路径</small>
        </label>
        {workflowType === "manga" && (
          <div className="preview-row">
            <button type="button" onClick={previewImages} disabled={busy || !path}>
              预览图片排序
            </button>
            {preview && <small>{preview}</small>}
          </div>
        )}
        {workflowType === "text" && <label>
          卷名匹配规则 <span>可选</span>
          <input
            value={volumeSplitters}
            onChange={(event) => setVolumeSplitters(event.target.value)}
            placeholder="第一卷，第一部，第一篇"
          />
          <small>可用逗号、斜杠、分号或空格分隔；数字会自动泛化</small>
        </label>}
        {workflowType === "text" && <label>
          章节名匹配规则 <span>可选</span>
          <input
            value={chapterSplitters}
            onChange={(event) => setChapterSplitters(event.target.value)}
            placeholder="第一章，第一话，序章"
          />
          <small>例如“第一章”会转换为“第&#123;num&#125;章”</small>
        </label>}
        {workflowType === "manga" && (
          <>
            <label>
              视觉模型
              <select disabled={!project?.vision_model_ready} value="qwen3.7-flash">
                <option value="qwen3.7-flash">Qwen3.7 Flash</option>
              </select>
              {!project?.vision_model_ready && (
                <small>未检测到 QWEN_API_KEY 或 DASHSCOPE_API_KEY</small>
              )}
            </label>
            <label>
              每批图片数
              <input
                type="number"
                min={1}
                max={10}
                value={imageBatchSize}
                onChange={(event) => setImageBatchSize(Number(event.target.value))}
              />
            </label>
            <label className="debug-option">
              <input
                type="checkbox"
                checked={ocrEnabled}
                disabled={!project?.ocr?.available}
                onChange={(event) => setOcrEnabled(event.target.checked)}
              />
              <span>
                <strong>使用 PaddleOCR-VL 修正对白</strong>
                <small>
                  {project?.ocr?.available
                    ? "OCR 与视觉模型并行运行"
                    : project?.ocr?.reason ?? "OCR 未配置"}
                </small>
              </span>
            </label>
          </>
        )}
        <fieldset className="mode-field">
          <legend>运行方式</legend>
          <button
            type="button"
            className={mode === "automatic" ? "selected" : ""}
            onClick={() => setMode("automatic")}
          >
            <strong>自动完成</strong>
            <small>连续执行全部阶段</small>
          </button>
          <button
            type="button"
            className={mode === "staged" ? "selected" : ""}
            onClick={() => setMode("staged")}
          >
            <strong>逐阶段确认</strong>
            <small>每个大阶段后暂停</small>
          </button>
        </fieldset>
        <label className="debug-option">
          <input
            type="checkbox"
            checked={debug}
            onChange={(event) => setDebug(event.target.checked)}
          />
          <span>
            <strong>调试模式</strong>
            <small>
              {workflowType === "manga"
                ? "仅处理排序后的前 5 个图片批次"
                : "剧情提取和对话提取分别最多处理 5 个 Chunk"}
            </small>
          </span>
        </label>
        {error && <p className="dialog-error">{error}</p>}
        <div className="dialog-actions">
          <button type="button" className="secondary" onClick={onClose}>
            取消
          </button>
          <button
            className="primary"
            disabled={busy || (workflowType === "manga" && !project?.vision_model_ready)}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Check size={16} />
            )}
            开始提取
          </button>
        </div>
      </form>
    </div>
  );
}
