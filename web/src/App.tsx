import {
  ArrowUp,
  BookOpenText,
  Check,
  ChevronRight,
  Database,
  FileText,
  FolderOpen,
  LoaderCircle,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Sparkles,
  X,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

type DatasetSummary = {
  directory: string;
  name: string;
  valid: boolean;
  characters?: number;
  conversations?: number;
};

type Project = {
  name: string;
  path: string;
  model_ready: boolean;
  datasets: DatasetSummary[];
};

type Message = { role: "user" | "assistant"; content: string };
type ExtractionState = {
  task: string;
  status: string;
  stage?: string;
  dataset_root?: string;
  error?: string;
};

const stageNames: Record<string, string> = {
  chunk: "整理原文",
  plot_extraction: "提取剧情",
  character_filter: "筛选角色",
  plot_reconstruction: "重建剧情",
  dialogue_extraction: "提取对话",
  character_profile: "生成角色档案",
  dataset: "生成数据集",
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? "请求失败");
  }
  return response.json() as Promise<T>;
}

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatSession, setChatSession] = useState<number>();
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [showExtraction, setShowExtraction] = useState(false);
  const [extraction, setExtraction] = useState<ExtractionState>();
  const [error, setError] = useState<string>();
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshProject = useCallback(async () => {
    try {
      setProject(await api<Project>("/api/project"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "无法连接本地服务");
    }
  }, []);

  useEffect(() => void refreshProject(), [refreshProject]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!extraction || ["completed", "failed", "incomplete"].includes(extraction.status)) {
      return;
    }
    const timer = window.setInterval(async () => {
      const state = await api<ExtractionState>(`/api/extractions/${extraction.task}`);
      setExtraction(state);
      if (state.status === "completed") {
        void refreshProject();
        setMessages((current) => [
          ...current,
          { role: "assistant", content: `数据集已经生成：${state.dataset_root}` },
        ]);
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [extraction, refreshProject]);

  async function sendMessage(content = draft) {
    const value = content.trim();
    if (!value || sending) return;
    setDraft("");
    setError(undefined);
    setMessages((current) => [...current, { role: "user", content: value }]);
    setSending(true);
    try {
      const response = await api<{ chat_session: number; content: string }>("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: value, chat_session: chatSession }),
      });
      setChatSession(response.chat_session);
      setMessages((current) => [...current, { role: "assistant", content: response.content }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "消息发送失败");
    } finally {
      setSending(false);
    }
  }

  async function confirmStage() {
    if (!extraction) return;
    await api(`/api/extractions/${extraction.task}/confirm`, { method: "POST" });
    setExtraction((current) => current ? { ...current, status: "running" } : current);
  }

  function newChat() {
    setMessages([]);
    setChatSession(undefined);
    setError(undefined);
  }

  const isEmpty = messages.length === 0;
  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="brand-row">
          <div className="brand-mark">R</div>
          <div className="brand-copy">
            <strong>RAR</strong>
            <span>Read And Retrieve</span>
          </div>
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)}>
            <X size={18} />
          </button>
        </div>
        <button className="new-chat" onClick={newChat}>
          <Plus size={17} /> 新对话
        </button>

        <div className="side-section">
          <div className="side-title"><FolderOpen size={14} /> 当前项目</div>
          <div className="project-card">
            <span className="project-dot" />
            <div><strong>{project?.name ?? "正在连接…"}</strong><small>{project?.path}</small></div>
          </div>
        </div>

        <div className="side-section datasets-section">
          <div className="side-title"><Database size={14} /> 数据集 <span>{project?.datasets.length ?? 0}</span></div>
          <div className="dataset-list">
            {project?.datasets.map((dataset) => (
              <button className="dataset-item" key={dataset.directory}>
                <BookOpenText size={16} />
                <span><strong>{dataset.name}</strong><small>{dataset.conversations ?? 0} 段对话 · {dataset.characters ?? 0} 个角色</small></span>
                <ChevronRight size={14} />
              </button>
            ))}
            {project?.datasets.length === 0 && <p className="empty-side">提取完成的数据集会出现在这里。</p>}
          </div>
        </div>

        <div className="sidebar-footer">
          <span className={`status-light ${project?.model_ready ? "ready" : ""}`} />
          {project?.model_ready ? "模型已连接" : "等待模型配置"}
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((value) => !value)}>
            {sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}
          </button>
          <div className="topbar-title"><span className="topbar-dot" /> 项目助手</div>
          <button className="extract-button" onClick={() => setShowExtraction(true)}>
            <Sparkles size={16} /> 提取数据集
          </button>
        </header>

        <section className={`conversation ${isEmpty ? "welcome-state" : ""}`}>
          {isEmpty ? (
            <div className="welcome">
              <div className="welcome-icon"><MessageSquareText size={27} /></div>
              <p className="eyebrow">LOCAL DATASET STUDIO</p>
              <h1>把故事，整理成<br /><em>可追溯的对话。</em></h1>
              <p className="welcome-copy">给我一个本地文本路径，或直接开始一次提取。剧情、角色档案、原始来源和训练数据会一起保存在当前项目中。</p>
              <div className="suggestions">
                <button onClick={() => setShowExtraction(true)}><FileText size={18} /><span><strong>从文本开始提取</strong><small>创建剧情与角色对话数据集</small></span><ChevronRight size={15} /></button>
                <button onClick={() => void sendMessage("检查当前项目中已有的数据集和中间产物。")}> <Database size={18} /><span><strong>检查当前项目</strong><small>了解资源与已有结果</small></span><ChevronRight size={15} /></button>
              </div>
            </div>
          ) : (
            <div className="message-stream">
              {messages.map((message, index) => (
                <article className={`message ${message.role}`} key={`${index}-${message.role}`}>
                  <div className="message-avatar">{message.role === "assistant" ? "R" : "你"}</div>
                  <div><span>{message.role === "assistant" ? "RAR" : "你"}</span><p>{message.content || "（无文本输出）"}</p></div>
                </article>
              ))}
              {sending && <article className="message assistant"><div className="message-avatar">R</div><div><span>RAR</span><p className="thinking"><i /><i /><i /></p></div></article>}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        {extraction && !["completed", "failed"].includes(extraction.status) && (
          <div className={`run-banner ${extraction.status}`}>
            {extraction.status === "completed" ? <Check size={17} /> : <LoaderCircle className="spin" size={17} />}
            <span><strong>{extraction.status === "incomplete" ? "提取需要处理" : "正在构建数据集"}</strong>{stageNames[extraction.stage ?? ""] ?? "准备资源"}</span>
            {extraction.status === "awaiting_confirmation" && <button className="confirm-button" onClick={() => void confirmStage()}>确认并继续</button>}
            {extraction.error && <small>{extraction.error}</small>}
          </div>
        )}
        {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError(undefined)}><X size={15} /></button></div>}

        <div className="composer-wrap">
          <form className="composer" onSubmit={(event) => { event.preventDefault(); void sendMessage(); }}>
            <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(); } }} placeholder="告诉 RAR 你想提取、检查或修改什么…" rows={1} />
            <button className="send-button" disabled={!draft.trim() || sending}><ArrowUp size={18} /></button>
          </form>
          <p className="composer-hint">Enter 发送 · Shift + Enter 换行 · 所有结果仅保存在本地</p>
        </div>
      </main>

      {showExtraction && <ExtractionDialog onClose={() => setShowExtraction(false)} onStarted={(state) => { setExtraction(state); setShowExtraction(false); setMessages((current) => [...current, { role: "user", content: `从 ${state.task} 对应的资源开始提取数据集。` }]); }} />}
    </div>
  );
}

function ExtractionDialog({ onClose, onStarted }: { onClose: () => void; onStarted: (state: ExtractionState) => void }) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [chapter, setChapter] = useState("");
  const [mode, setMode] = useState<"automatic" | "staged">("automatic");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(undefined);
    try {
      const state = await api<ExtractionState>("/api/extractions", {
        method: "POST",
        body: JSON.stringify({ mode, manifest: { name, meta: {}, resources: [{ path, resource_type: "text", display_name: chapter || name, narrative_order: 0, meta: chapter ? { chapter } : {} }] } }),
      });
      onStarted(state);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "无法开始提取");
    } finally {
      setBusy(false);
    }
  }

  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <form className="dialog" onSubmit={submit}>
      <div className="dialog-heading"><div className="dialog-icon"><Sparkles size={20} /></div><div><p>NEW DATASET</p><h2>从文本提取对话</h2></div><button type="button" className="icon-button" onClick={onClose}><X size={18} /></button></div>
      <p className="dialog-copy">RAR 会依次整理原文、识别剧情、筛选角色、提取对白，并生成角色档案与 ShareGPT 数据。</p>
      <label>数据集名称<input required value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：我的青春恋爱物语" /></label>
      <label>文本文件路径<input required value={path} onChange={(event) => setPath(event.target.value)} placeholder="resources/book.txt" /><small>填写当前项目内的相对路径</small></label>
      <label>章节或卷名 <span>可选</span><input value={chapter} onChange={(event) => setChapter(event.target.value)} placeholder="第一卷 · 第一章" /></label>
      <fieldset className="mode-field"><legend>运行方式</legend><button type="button" className={mode === "automatic" ? "selected" : ""} onClick={() => setMode("automatic")}><strong>自动完成</strong><small>连续执行全部阶段</small></button><button type="button" className={mode === "staged" ? "selected" : ""} onClick={() => setMode("staged")}><strong>逐阶段确认</strong><small>每个大阶段后暂停</small></button></fieldset>
      {error && <p className="dialog-error">{error}</p>}
      <div className="dialog-actions"><button type="button" className="secondary" onClick={onClose}>取消</button><button className="primary" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} 开始提取</button></div>
    </form>
  </div>;
}
