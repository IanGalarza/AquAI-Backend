import json
from pathlib import Path

from pydantic import JsonValue


class FileTabularRuleCatalog:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(__file__).parents[1] / "config" / "tabular_rules.json"

    def load(self) -> dict[str, JsonValue]:
        with self._path.open(encoding="utf-8") as stream:
            return json.load(stream)
