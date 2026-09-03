import asyncio

from rar_agent.models.base import ModelMessage, ModelRequest, ModelResponse
from rar_agent.models.scheduler import ModelScheduler


class ControlledClient:
    provider = "test"

    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ModelResponse(content=request.messages[-1].content)


class BlockingClient:
    provider = "test"

    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.active_by_kind = {"workflow": 0, "chat": 0}
        self.eight_workflow_started = asyncio.Event()
        self.twelve_calls_started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        content = request.messages[-1].content or ""
        kind = "chat" if content.startswith("chat") else "workflow"
        self.active += 1
        self.active_by_kind[kind] += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.active_by_kind["workflow"] == 8:
            self.eight_workflow_started.set()
        if self.active == 12:
            self.twelve_calls_started.set()
        await self.release.wait()
        self.active -= 1
        self.active_by_kind[kind] -= 1
        return ModelResponse(content=content)


def _request(name: str) -> ModelRequest:
    return ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content=name)],
    )


def test_scheduler_defaults_to_twelve_global_and_eight_per_workflow() -> None:
    scheduler = ModelScheduler()

    assert scheduler.default_limit == 12
    assert scheduler.default_workflow_limit == 8


async def test_scheduler_applies_one_global_model_limit() -> None:
    client = ControlledClient()
    scheduler = ModelScheduler(default_limit=2)

    await asyncio.gather(
        *(
            scheduler.complete(
                client,
                _request(str(index)),
                workload="workflow",
                workload_id=f"workflow-{index % 3}",
            )
            for index in range(9)
        )
    )

    assert client.maximum_active == 2


async def test_scheduler_limits_one_workflow_to_eight_calls() -> None:
    client = BlockingClient()
    scheduler = ModelScheduler(default_limit=12, default_workflow_limit=8)
    tasks = [
        asyncio.create_task(
            scheduler.complete(
                client,
                _request(f"workflow-{index}"),
                workload="workflow",
                workload_id="workflow-a",
            )
        )
        for index in range(12)
    ]

    try:
        await asyncio.wait_for(client.eight_workflow_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert client.active == 8
        assert client.active_by_kind["workflow"] == 8
    finally:
        client.release.set()
        await asyncio.gather(*tasks)


async def test_harness_uses_capacity_left_by_workflow() -> None:
    client = BlockingClient()
    scheduler = ModelScheduler(default_limit=12, default_workflow_limit=8)
    workflow_tasks = [
        asyncio.create_task(
            scheduler.complete(
                client,
                _request(f"workflow-{index}"),
                workload="workflow",
                workload_id="workflow-a",
            )
        )
        for index in range(12)
    ]
    await asyncio.wait_for(client.eight_workflow_started.wait(), timeout=1)
    chat_tasks = [
        asyncio.create_task(
            scheduler.complete(
                client,
                _request(f"chat-{index}"),
                workload="chat",
                workload_id=f"chat-{index}",
            )
        )
        for index in range(4)
    ]

    try:
        await asyncio.wait_for(client.twelve_calls_started.wait(), timeout=1)
        assert client.active == 12
        assert client.active_by_kind == {"workflow": 8, "chat": 4}
    finally:
        client.release.set()
        await asyncio.gather(*workflow_tasks, *chat_tasks)


class OrderedClient:
    provider = "test"

    def __init__(self) -> None:
        self.order: list[str] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        name = request.messages[-1].content or ""
        self.order.append(name)
        if name == "first":
            self.first_started.set()
            await self.release_first.wait()
        return ModelResponse(content=name)


async def test_scheduler_runs_waiting_chat_before_workflow() -> None:
    client = OrderedClient()
    scheduler = ModelScheduler(default_limit=1)
    first = asyncio.create_task(
        scheduler.complete(
            client,
            _request("first"),
            workload="workflow",
            workload_id="workflow-a",
        )
    )
    await client.first_started.wait()
    workflow = asyncio.create_task(
        scheduler.complete(
            client,
            _request("workflow"),
            workload="workflow",
            workload_id="workflow-b",
        )
    )
    await asyncio.sleep(0)
    chat = asyncio.create_task(
        scheduler.complete(
            client,
            _request("chat"),
            workload="chat",
            workload_id="chat-1",
        )
    )
    await asyncio.sleep(0)

    client.release_first.set()
    await asyncio.gather(first, workflow, chat)

    assert client.order == ["first", "chat", "workflow"]
