# AquAI Backend

AquAI es una aplicación con backend en Python, prevista para utilizar FastAPI,
PostgreSQL, Qdrant y LangGraph, y procesar archivos CSV, Excel y PDF.

## Arquitectura

El código está organizado en `src/aquai`, siguiendo una separación inspirada en
Clean Architecture:

- `presentation`: entrada de la aplicación, con controllers y schemas de FastAPI.
- `application`: casos de uso, servicios, interfaces, DTOs, mappers y utilidades.
  Incluye `agents` con `graphs`, `nodes`, `state` y `prompts` para futuros agentes.
- `domain`: modelos, enums, eventos y objetos de valor, sin dependencias de infraestructura.
- `infrastructure`: implementaciones concretas, commands, queries, persistencia,
  proveedores, migraciones y configuración. Incluye `ai` con `tools`, `memory` y
  `checkpoints`, y `processing` con `tabular` y `documents`.

Los tests se separan en `tests/unit`, `tests/integration` y `tests/evaluation`.
Esta última carpeta se reserva para benchmarks y evaluaciones de los modelos de
lenguaje y del sistema.
