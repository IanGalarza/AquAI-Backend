from functools import lru_cache

from aquai.application.use_cases.evaluate_tabular_file import EvaluateTabularFile
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.llm.ollama import OllamaProvider


@lru_cache(maxsize=1)
def get_tabular_settings() -> TabularSettings:
    return TabularSettings()


def get_tabular_evaluator() -> EvaluateTabularFile:
    settings = get_tabular_settings()
    processor = PandasTabularProcessor(
        max_cells=settings.max_cells,
        max_sheets=settings.max_sheets,
        max_expanded_bytes=settings.max_expanded_bytes,
        sample_rows=settings.sample_rows,
    )
    return EvaluateTabularFile(
        processor,
        OllamaProvider(settings),
        default_model=settings.default_model,
        max_upload_bytes=settings.max_upload_bytes,
    )
