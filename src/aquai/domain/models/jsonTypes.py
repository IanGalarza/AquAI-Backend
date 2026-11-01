from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

JsonDocument: TypeAlias = str | int | float | bool | list[JsonValue] | dict[str, JsonValue]
