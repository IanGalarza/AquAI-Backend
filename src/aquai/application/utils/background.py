import asyncio
from collections.abc import Callable
from typing import Any


async def complete_write(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Una cancelación de asyncio no detiene el thread ni su transacción.
        await asyncio.shield(task)
        raise
