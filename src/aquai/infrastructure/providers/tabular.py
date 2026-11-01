from functools import lru_cache

from aquai.infrastructure.config.tabular import TabularSettings


#
# Devuelve la configuración del procesamiento tabular, cacheada.
#
@lru_cache
def get_tabular_settings() -> TabularSettings:
    return TabularSettings()
