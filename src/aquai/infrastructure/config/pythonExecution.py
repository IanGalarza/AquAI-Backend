"""Presupuestos de herramientas Python; no amplían los límites del SQL Agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PythonExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PYTHON_EXECUTION_", extra="ignore")
    max_input_rows: int = Field(default=1000, ge=1, le=10000)
    max_input_bytes: int = Field(default=1048576, ge=1024, le=10485760)
    max_cells: int = Field(default=20000, ge=1, le=100000)
    max_memory_bytes: int = Field(default=16777216, ge=1048576, le=67108864)
    max_duration_ms: int = Field(default=10000, ge=1, le=60000)
    max_result_rows: int = Field(default=1000, ge=1, le=10000)
    max_result_bytes: int = Field(default=1048576, ge=8192, le=10485760)
    max_chart_points: int = Field(default=1000, ge=1, le=10000)
    max_chart_series: int = Field(default=50, ge=1, le=100)
    decimal_precision: int = Field(default=38, ge=28, le=64)
