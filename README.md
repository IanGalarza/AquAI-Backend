# AquAI Backend

Backend para cargar originales CSV/XLSX, preparar tablas conservadoramente y
guardar todos sus registros en PostgreSQL. Qwen describe los conjuntos y Ollama
genera embeddings con otro modelo; Qdrant guarda el índice semántico derivado.

La arquitectura mantiene Domain, Application, Infrastructure y Presentation.
Las entidades de conversación siguen disponibles para una etapa posterior.

## Instalación local

Requiere Python 3.11+, Docker con Compose y Ollama instalado en el host.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Configurar las credenciales de PostgreSQL en `.env`. `DATABASE_URL` debe
corresponder al mismo servicio que las variables `POSTGRES_*`.
Codificar los caracteres especiales del usuario/contraseña en la URL.

```powershell
docker compose up -d postgres qdrant
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
ollama serve
```

Si Ollama ya está ejecutándose, basta instalar los modelos.
Desde otra terminal:

```powershell
python -m alembic upgrade head
python -m uvicorn aquai.main:app --reload
```

La migración inicial consolidada corresponde a una base nueva o a la base local
vacía reconstruida durante este refactor. No aplicar esta historia sobre una
base anterior con registros: requiere una migración de datos específica.

Documentación interactiva: [Swagger local](http://localhost:8000/docs).

## API de fuentes

| Método | Ruta | Función |
|---|---|---|
| POST | /sources | Multipart con un único campo file; guarda el original y responde 202. |
| GET | /sources | Lista paginada; page, page_size, name y processing_status. |
| GET | /sources/{id} | Estados, advertencias y datasets con columnas y conteos. |
| PATCH | /sources/{id} | JSON con isActive booleano; cambia la activación. |

SHA-256 identifica originales binariamente iguales: una carga repetida devuelve
el mismo ID y `duplicate=true`. El detalle no incluye binarios ni filas completas.
`queryable` exige un Dataset confirmado y una Source y Dataset activos; activar
una carga fallida no la hace consultable.

La cola vive en PostgreSQL. El worker confirma SQL antes de llamar a Qwen.
Un error semántico conserva las tablas y queda visible en el reporte.
Un error de embeddings/Qdrant conserva SQL y marca el índice como fallido.

## Reindexación y pruebas

Reindexar desde las descripciones guardadas, sin volver a leer el original:

```powershell
python -m aquai.infrastructure.workers.reindexSources SOURCE_UUID
```

El backend debe estar iniciado para que el worker complete esa solicitud.
Si se cambia de modelo de embeddings, configurar otra colección y reindexar.

```powershell
python -m unittest discover -s tests -v
python -m alembic check
git diff --check
```

La suite es offline: archivos sintéticos, SQLite, HTTP simulado y Qdrant en memoria.

Los límites y URLs están centralizados en `TabularSettings` y `.env.example`.
CSV conserva códigos, marcadores y valores ambiguos como texto. XLSX conserva
fórmulas sin calcularlas y publica sólo regiones con evidencia tabular suficiente.

Ver [overview técnico](docs/ingestionOverview.md) e
[informe del refactor](simpleIngestionNotes.md).
