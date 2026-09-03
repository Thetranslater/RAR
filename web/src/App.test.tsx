import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import App from "./App";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const project = {
  name: "test-project",
  path: "D:/test-project",
  model_ready: true,
};

describe("App chat navigation", () => {
  beforeEach(() => {
    window.localStorage.clear();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("creates a chat lazily and renders its background response", async () => {
    let created = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats" || path === "/api/chats?archived=true") {
          return jsonResponse(
            path.includes("archived") || !created
              ? []
              : [
                  {
                    id: 1,
                    title: "你好",
                    created_at: new Date().toISOString(),
                    updated_at: new Date().toISOString(),
                    archived_at: null,
                  },
                ],
          );
        }
        if (path === "/api/chat" && init?.method === "POST") {
          created = true;
          return jsonResponse(
            { chat_session: 1, run_id: "run-1", status: "queued" },
            202,
          );
        }
        if (path === "/api/chats/1/runs/run-1") {
          return jsonResponse({
            chat_session: 1,
            run_id: "run-1",
            status: "completed",
            content: "你好！",
          });
        }
        if (path === "/api/chats/1/messages?limit=100") {
          return jsonResponse({
            items: created
              ? [
                  {
                    id: 1,
                    role: "user",
                    content: "你好",
                    kind: "message",
                    created_at: new Date().toISOString(),
                    tool_calls: [],
                  },
                  {
                    id: 2,
                    role: "assistant",
                    content: "你好！",
                    kind: "message",
                    created_at: new Date().toISOString(),
                    tool_calls: [],
                  },
                ]
              : [],
            next_before: null,
          });
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/chats/new"]}>
        <App />
      </MemoryRouter>,
    );
    const composer = await screen.findByPlaceholderText(
      "告诉 RAR 你想提取、检查或修改什么...",
    );
    await user.type(composer, "你好{Enter}");

    await screen.findByText("你好！");
    expect(screen.getAllByText("你好").length).toBeGreaterThan(0);
    expect(screen.getByText("test-project")).not.toBeNull();
  });

  it("restores a pending approval and resumes the same run", async () => {
    let approved = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats") {
          return jsonResponse([
            {
              id: 7,
              title: "创建文件",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
              run_id: approved ? null : "run-7",
              run_status: approved ? null : "awaiting_approval",
            },
          ]);
        }
        if (path === "/api/chats?archived=true") return jsonResponse([]);
        if (path === "/api/chats/7/messages?limit=100") {
          return jsonResponse({
            items: approved
              ? [
                  {
                    id: 1,
                    role: "assistant",
                    content: "文件已经创建。",
                    kind: "message",
                    created_at: new Date().toISOString(),
                    tool_calls: [],
                  },
                ]
              : [],
            next_before: null,
          });
        }
        if (path === "/api/chats/7/runs/run-7") {
          return jsonResponse({
            chat_session: 7,
            run_id: "run-7",
            status: approved ? "completed" : "awaiting_approval",
            approval: approved
              ? null
              : {
                  token: "approval-token",
                  expires_in_seconds: 300,
                  tools: [
                    {
                      name: "write_file",
                      description: "Create a file",
                      risk: "medium",
                      effect: "write",
                      reason: "Creates a file",
                      arguments: { path: "approved.txt" },
                    },
                  ],
                },
          });
        }
        if (
          path === "/api/approvals/approval-token" &&
          init?.method === "POST"
        ) {
          approved = true;
          return jsonResponse(
            { chat_session: 7, run_id: "run-7", status: "queued" },
            202,
          );
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/chats/7"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByText("RAR 请求执行一项受保护操作");
    expect(screen.getByText("approved.txt")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "允许一次" }));

    await waitFor(() => expect(screen.getByText("文件已经创建。")).not.toBeNull());
    expect(screen.queryByText("RAR 请求执行一项受保护操作")).toBeNull();
  });

  it("renders a chat action menu outside the scrolling chat list", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats") {
          return jsonResponse([
            {
              id: 3,
              title: "Only chat",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
            },
          ]);
        }
        if (path === "/api/chats?archived=true") return jsonResponse([]);
        if (path === "/api/chats/3/messages?limit=100") {
          return jsonResponse({ items: [], next_before: null });
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    const user = userEvent.setup();
    const { container } = render(
      <MemoryRouter initialEntries={["/chats/3"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findAllByText("Only chat");
    const menuButton = container.querySelector<HTMLButtonElement>(
      ".chat-menu-button",
    );
    expect(menuButton).not.toBeNull();
    await user.click(menuButton!);

    expect(container.querySelector(".chat-list .chat-menu")).toBeNull();
    expect(document.body.querySelector(".chat-menu")).not.toBeNull();
  });

  it("does not reload messages when an idle chat-list poll completes", async () => {
    let messageRequests = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats") {
          return jsonResponse([
            {
              id: 9,
              title: "Stable chat",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
            },
          ]);
        }
        if (path === "/api/chats?archived=true") return jsonResponse([]);
        if (path === "/api/chats/9/messages?limit=100") {
          messageRequests += 1;
          return jsonResponse({ items: [], next_before: null });
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    render(
      <MemoryRouter initialEntries={["/chats/9"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findAllByText("Stable chat");
    await new Promise((resolve) => window.setTimeout(resolve, 1_700));

    expect(messageRequests).toBe(1);
  });

  it("disables the composer while the selected chat workflow is running", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats") {
          return jsonResponse([
            {
              id: 12,
              title: "Running extraction",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
              extraction_task: "extract-12",
              extraction_status: "running",
            },
          ]);
        }
        if (path === "/api/chats?archived=true") return jsonResponse([]);
        if (path === "/api/chats/12/messages?limit=100") {
          return jsonResponse({ items: [], next_before: null });
        }
        if (path === "/api/extractions/extract-12") {
          return jsonResponse({
            task: "extract-12",
            chat_session: 12,
            status: "running",
            stage: "plot_extraction",
          });
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    render(
      <MemoryRouter initialEntries={["/chats/12"]}>
        <App />
      </MemoryRouter>,
    );

    const composer = await screen.findByPlaceholderText(
      "数据集提取正在运行，阶段完成后可继续对话",
    );
    expect((composer as HTMLTextAreaElement).disabled).toBe(true);
    expect(document.querySelector(".composer.locked")).not.toBeNull();
  });

  it("enables chat at a staged pause and disables confirmation during its Agent run", async () => {
    let agentStarted = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats") {
          return jsonResponse([
            {
              id: 13,
              title: "Paused extraction",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              archived_at: null,
              run_id: agentStarted ? "run-13" : null,
              run_status: agentStarted ? "queued" : null,
              extraction_task: "extract-13",
              extraction_status: "awaiting_confirmation",
            },
          ]);
        }
        if (path === "/api/chats?archived=true") return jsonResponse([]);
        if (path === "/api/chats/13/messages?limit=100") {
          return jsonResponse({ items: [], next_before: null });
        }
        if (path === "/api/extractions/extract-13") {
          return jsonResponse({
            task: "extract-13",
            chat_session: 13,
            status: "awaiting_confirmation",
            stage: "plot_extraction",
          });
        }
        if (path === "/api/chat" && init?.method === "POST") {
          agentStarted = true;
          return jsonResponse(
            { chat_session: 13, run_id: "run-13", status: "queued" },
            202,
          );
        }
        if (path === "/api/chats/13/runs/run-13") {
          return jsonResponse({
            chat_session: 13,
            run_id: "run-13",
            status: "queued",
          });
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/chats/13"]}>
        <App />
      </MemoryRouter>,
    );
    await screen.findByText("提取剧情已完成");
    const composer = screen.getByPlaceholderText(
      "告诉 RAR 你想提取、检查或修改什么...",
    ) as HTMLTextAreaElement;
    expect(composer.disabled).toBe(false);
    await user.type(composer, "修正角色名称{Enter}");

    const confirm = screen.getByRole("button", {
      name: "确认并继续",
    }) as HTMLButtonElement;
    await waitFor(() => expect(confirm.disabled).toBe(true));
    expect(
      screen.getByPlaceholderText("当前对话正在执行任务"),
    ).not.toBeNull();
  });

  it("submits separate normalized volume and chapter splitters", async () => {
    let extractionBody: Record<string, unknown> | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path === "/api/project") return jsonResponse(project);
        if (path === "/api/chats" || path === "/api/chats?archived=true") {
          return jsonResponse([]);
        }
        if (path === "/api/extractions" && init?.method === "POST") {
          extractionBody = JSON.parse(String(init.body));
          return jsonResponse(
            { task: "extract-1", status: "queued", chat_session: 11 },
            202,
          );
        }
        throw new Error("Unexpected request: " + path);
      }),
    );

    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/chats/new"]}>
        <App />
      </MemoryRouter>,
    );
    await user.click(await screen.findByRole("button", { name: "提取数据集" }));
    await user.type(screen.getByPlaceholderText("例如：我的青春恋爱物语"), "测试作品");
    await user.type(screen.getByPlaceholderText("resources/book.txt"), "book.txt");
    await user.type(
      screen.getByPlaceholderText("第一卷，第一部，第一篇"),
      "第一卷，第一部/第二卷",
    );
    await user.type(
      screen.getByPlaceholderText("第一章，第一话，序章"),
      "第一章；序章 第一话",
    );
    await user.click(screen.getByRole("button", { name: "开始提取" }));

    await waitFor(() => expect(extractionBody).toBeDefined());
    expect(extractionBody?.volume_splitters).toEqual([
      "第{num}卷",
      "第{num}部",
    ]);
    expect(extractionBody?.chapter_splitters).toEqual([
      "第{num}章",
      "序章",
      "第{num}话",
    ]);
    expect(extractionBody?.manifest).toEqual({
      name: "测试作品",
      meta: {},
      resources: [
        {
          path: "book.txt",
          resource_type: "text",
          display_name: "测试作品",
          narrative_order: 0,
          meta: {},
        },
      ],
    });
  });
});
