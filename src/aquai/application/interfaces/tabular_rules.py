from typing import Protocol

from pydantic import JsonValue


class TabularRuleCatalog(Protocol):
    def load(self) -> dict[str, JsonValue]: ...
