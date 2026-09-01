import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("App chat", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(() => ({ animation: "scheduled" })),
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);
        if (path === "/api/project") {
          return jsonResponse({
            name: "test-project",
            path: "D:/test-project",
            model_ready: true,
            datasets: [],
          });
        }
        if (path === "/api/chat") {
          await new Promise((resolve) => window.setTimeout(resolve, 0));
          return jsonResponse({ chat_session: 1, content: "你好！", tool_calls: 0 });
        }
        throw new Error(`Unexpected request: ${path}`);
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("keeps the application mounted after a user and assistant message", async () => {
    const user = userEvent.setup();
    render(<App />);
    const composer = await screen.findByPlaceholderText(
      "告诉 RAR 你想提取、检查或修改什么…",
    );

    await user.type(composer, "你好{Enter}");

    await waitFor(() => expect(screen.getByText("你好！")).not.toBeNull());
    expect(screen.getByText("项目助手")).not.toBeNull();
  });
});
