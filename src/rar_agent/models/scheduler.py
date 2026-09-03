"""Process-local fair concurrency scheduling shared by Chat and Workflow runs."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal

from rar_agent.models.base import ModelClient, ModelRequest, ModelResponse

WorkloadKind = Literal["chat", "workflow"]
DEFAULT_MODEL_CONCURRENCY = 12
DEFAULT_WORKFLOW_CONCURRENCY = 8


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[None]
    workload: WorkloadKind
    workload_id: str
    workflow_limit: int
    granted: bool = False


class ModelScheduler:
    """Share one model pool while limiting each individual Workflow."""

    def __init__(
        self,
        *,
        default_limit: int = DEFAULT_MODEL_CONCURRENCY,
        default_workflow_limit: int = DEFAULT_WORKFLOW_CONCURRENCY,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be positive")
        if default_workflow_limit < 1:
            raise ValueError("default_workflow_limit must be positive")
        self.default_limit = default_limit
        self.default_workflow_limit = default_workflow_limit
        self._active = 0
        self._workflow_active: dict[str, int] = {}
        self._guard = asyncio.Lock()
        self._chat_waiters: deque[_Waiter] = deque()
        self._workflow_waiters: dict[str, deque[_Waiter]] = {}
        self._workflow_order: deque[str] = deque()

    async def complete(
        self,
        client: ModelClient,
        request: ModelRequest,
        *,
        workload: WorkloadKind = "workflow",
        workload_id: str | None = None,
        workload_limit: int | None = None,
    ) -> ModelResponse:
        if workload_limit is not None and workload_limit < 1:
            raise ValueError("workload_limit must be positive")
        resolved_workload_id = workload_id or workload
        await self._acquire(
            workload,
            resolved_workload_id,
            workload_limit or self.default_workflow_limit,
        )
        try:
            return await client.complete(request)
        finally:
            await self._release(workload, resolved_workload_id)

    async def _acquire(
        self,
        workload: WorkloadKind,
        workload_id: str,
        workflow_limit: int,
    ) -> None:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            future=loop.create_future(),
            workload=workload,
            workload_id=workload_id,
            workflow_limit=workflow_limit,
        )
        async with self._guard:
            if self._can_start(waiter) and not self._has_waiters():
                self._mark_active(waiter)
                return
            if workload == "chat":
                self._chat_waiters.append(waiter)
            else:
                queue = self._workflow_waiters.get(workload_id)
                if queue is None:
                    queue = deque()
                    self._workflow_waiters[workload_id] = queue
                    self._workflow_order.append(workload_id)
                queue.append(waiter)
            self._dispatch()
        try:
            await waiter.future
        except BaseException:
            async with self._guard:
                if waiter.granted:
                    self._mark_released(waiter.workload, waiter.workload_id)
                else:
                    waiter.future.cancel()
                self._dispatch()
            raise

    async def _release(self, workload: WorkloadKind, workload_id: str) -> None:
        async with self._guard:
            self._mark_released(workload, workload_id)
            self._dispatch()

    def _has_waiters(self) -> bool:
        return bool(self._chat_waiters or self._workflow_order)

    def _dispatch(self) -> None:
        while self._active < self.default_limit:
            waiter = self._next_waiter()
            if waiter is None:
                return
            self._mark_active(waiter)
            waiter.future.set_result(None)

    def _next_waiter(self) -> _Waiter | None:
        while self._chat_waiters:
            waiter = self._chat_waiters.popleft()
            if not waiter.future.done():
                return waiter

        workflow_count = len(self._workflow_order)
        for _ in range(workflow_count):
            workload_id = self._workflow_order.popleft()
            queue = self._workflow_waiters[workload_id]
            while queue and queue[0].future.done():
                queue.popleft()
            if not queue:
                del self._workflow_waiters[workload_id]
                continue

            waiter = queue[0]
            if self._can_start(waiter):
                queue.popleft()
            if queue:
                self._workflow_order.append(workload_id)
            else:
                del self._workflow_waiters[workload_id]
            if self._can_start(waiter):
                return waiter
        return None

    def _can_start(self, waiter: _Waiter) -> bool:
        if self._active >= self.default_limit:
            return False
        if waiter.workload == "chat":
            return True
        return (
            self._workflow_active.get(waiter.workload_id, 0)
            < waiter.workflow_limit
        )

    def _mark_active(self, waiter: _Waiter) -> None:
        self._active += 1
        waiter.granted = True
        if waiter.workload == "workflow":
            self._workflow_active[waiter.workload_id] = (
                self._workflow_active.get(waiter.workload_id, 0) + 1
            )

    def _mark_released(self, workload: WorkloadKind, workload_id: str) -> None:
        self._active -= 1
        if workload != "workflow":
            return
        active = self._workflow_active[workload_id] - 1
        if active:
            self._workflow_active[workload_id] = active
        else:
            del self._workflow_active[workload_id]
