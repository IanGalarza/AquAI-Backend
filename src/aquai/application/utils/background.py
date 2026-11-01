import asyncio
from collections.abc import Callable
from typing import Any


#
# Ejecuta una escritura en un hilo y la deja terminar aunque se cancele la petición.
#
async def complete_write(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise
