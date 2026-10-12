"""Configuración compartida del lector y los adaptadores locales."""
from functools import lru_cache
from aquai.infrastructure.config.tabular import TabularSettings


@lru_cache
def get_tabular_settings() -> TabularSettings:
    return TabularSettings()
