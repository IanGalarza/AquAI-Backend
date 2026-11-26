# AquAI Backend

## Descripción general

API de AquAI, un asistente que responde preguntas sobre calidad del agua. Carga y procesa las
fuentes (planillas CSV y XLSX, documentos PDF y DOCX), gestiona las conversaciones y genera las
respuestas con modelos de lenguaje que se ejecutan localmente en Ollama. La interfaz web está en
AquAI-Frontend.

## Tecnologías utilizadas

| Área | Tecnologías |
|---|---|
| Lenguaje | Python 3.11 |
| API | FastAPI, Uvicorn, Pydantic |
| Persistencia | PostgreSQL 18, SQLAlchemy, Alembic |
| Búsqueda | Qdrant |
| Agentes | LangGraph, Ollama |
| Datos y documentos | pandas, Plotly, Docling, RapidOCR |
| Servicios locales | Docker Compose |
| Pruebas | unittest |

## Arquitectura

El código sigue Clean Architecture y se divide en cuatro capas dentro de `src/aquai`:

| Capa | Responsabilidad |
|---|---|
| `domain` | Entidades (fuentes, datasets, documentos, conversaciones y mensajes) y sus estados. |
| `application` | Casos de uso, servicios, reglas de interpretación, prompts de los agentes e interfaces que implementa la infraestructura. No depende de FastAPI, SQLAlchemy, LangGraph ni de los clientes de Qdrant y Ollama. |
| `infrastructure` | Persistencia y migraciones, procesamiento de archivos, clientes de Qdrant y Ollama, grafo de LangGraph, workers y composición de dependencias. |
| `presentation` | API HTTP: routers de FastAPI, traducción de errores y middlewares. |

Los controladores de `presentation` invocan los casos de uso, que `infrastructure/providers` arma
con las implementaciones de la infraestructura.

## Estructura del proyecto

```text
AquAI-Backend/
├── alembic.ini                 Configuración de Alembic
├── docker-compose.yml          PostgreSQL y Qdrant
├── pyproject.toml              Dependencias del paquete
├── .env.example                Variables de entorno de ejemplo
├── src/aquai/
│   ├── main.py                 Aplicación FastAPI
│   ├── domain/
│   │   ├── enums/              Estados y tipos del dominio
│   │   └── models/             Entidades
│   ├── application/
│   │   ├── useCases/           Casos de uso, divididos en Commands y Queries
│   │   ├── Services/           Servicios de los casos de uso, el grafo y los workers
│   │   ├── planning/           Validación y reparación de planes
│   │   ├── answering/          Construcción y verificación de respuestas
│   │   ├── conversations/      Contexto y presentación de las conversaciones
│   │   ├── Rules/              Reglas de interpretación (unidades, variables, idioma)
│   │   ├── prompts/            Prompts de los agentes
│   │   ├── contracts/          Contratos entre etapas y de conversaciones
│   │   ├── dtos/               Solicitudes y respuestas HTTP de fuentes
│   │   ├── interfaces/         Interfaces que implementa la infraestructura
│   │   ├── validation/         Validación de gráficos
│   │   └── utils/              Utilidades
│   ├── infrastructure/
│   │   ├── config/             Configuración leída de .env
│   │   ├── persistence/        Mapeo de SQLAlchemy y sesiones
│   │   ├── migrations/         Migraciones de Alembic
│   │   ├── commands/           Escrituras en PostgreSQL
│   │   ├── queries/            Lecturas y agente SQL
│   │   ├── analysis/           Agente Python y gráficos Plotly
│   │   ├── processing/         Lectura y conversión de planillas y documentos
│   │   ├── graphs/             Grafo de LangGraph
│   │   ├── providers/          Composición de dependencias, índices y clientes de Ollama
│   │   └── workers/            Workers de fuentes y scripts de mantenimiento
│   └── presentation/           Dependencias, errores HTTP y middlewares
│       └── controllers/        Routers de /sources y /conversations
└── tests/
    ├── unit/                   Pruebas unitarias
    └── integration/            Pruebas de integración de la API
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/sources` | Carga un archivo CSV, XLSX, PDF o DOCX (campo `file`). |
| GET | `/sources` | Lista las fuentes con paginación, filtros y orden. |
| GET | `/sources/{id}` | Devuelve el estado y el detalle de una fuente. |
| PATCH | `/sources/{id}` | Activa o desactiva una fuente. |
| PUT | `/sources/{id}/file` | Reemplaza el archivo de un documento PDF o DOCX. |
| DELETE | `/sources/{id}` | Elimina un documento PDF o DOCX. |
| POST | `/conversations` | Crea una conversación. |
| GET | `/conversations` | Lista las conversaciones. |
| GET | `/conversations/{id}` | Devuelve una conversación. |
| PATCH | `/conversations/{id}` | Renombra una conversación. |
| DELETE | `/conversations/{id}` | Elimina una conversación y sus mensajes. |
| GET | `/conversations/{id}/messages` | Devuelve el historial de mensajes. |
| POST | `/conversations/{id}/messages` | Envía una pregunta y devuelve la respuesta al terminar. |
| POST | `/conversations/{id}/messages/stream` | Envía una pregunta y transmite el progreso y la respuesta por SSE. |
| GET | `/conversations/{id}/messages/{message_id}/events` | Retoma el seguimiento de una respuesta en curso. |
| POST | `/conversations/{id}/messages/{message_id}/cancel` | Cancela una respuesta en curso. |

Swagger UI, en `http://localhost:8000/docs`, describe cada solicitud y respuesta. Las escrituras
(POST, PUT, PATCH y DELETE) solo se aceptan desde el origen del frontend o sin encabezado `Origin`,
como las de `curl.exe`; desde Swagger UI se rechazan con 403.

## Requisitos previos

- Python 3.11 o superior.
- Docker con Docker Compose.
- Ollama con los modelos configurados en `.env`.
- Conexión a internet la primera vez que se procesa un documento, para descargar los modelos de
  Docling y el tokenizador `BAAI/bge-m3`.

## Instalación y configuración

Los comandos se ejecutan en PowerShell desde la raíz del repositorio, donde se lee el archivo `.env`.

### 1. Crear el entorno virtual e instalar las dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. Configurar las variables de entorno

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`.env.example` documenta cada variable con su valor por defecto. Si se cambian las credenciales de
`POSTGRES_*`, actualizar también `DATABASE_URL`; los caracteres especiales se codifican (por ejemplo,
`@` como `%40`).

### 3. Iniciar PostgreSQL y Qdrant

```powershell
docker compose up -d postgres qdrant
```

### 4. Descargar los modelos de Ollama

Con Ollama en ejecución, descargar los modelos definidos en `.env`. Con la configuración por defecto:

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
ollama pull bge-m3
```

## Migraciones

Las migraciones se gestionan con Alembic y están en `src/aquai/infrastructure/migrations/versions/`.
Con el entorno virtual activo y PostgreSQL en ejecución:

```powershell
# Aplicar las migraciones pendientes
python -m alembic upgrade head

# Generar una migración a partir de los cambios en el mapeo de SQLAlchemy
python -m alembic revision --autogenerate -m "descripcion del cambio"
```

La migración generada debe revisarse antes de aplicarla. `--autogenerate` ignora las tablas
`dataset_<id>`, que se crean al procesar las planillas.

## Ejecución

Con PostgreSQL, Qdrant y Ollama en ejecución y las migraciones aplicadas:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn aquai.main:app
```

La API queda disponible en `http://localhost:8000`. No usar `--workers` con un valor mayor que 1:
las respuestas en curso se siguen en la memoria del proceso.

## Testing

Con el entorno virtual activo:

```powershell
# Todas las pruebas
python -m unittest discover -s tests -t .

# Un módulo, con el detalle de cada prueba
python -m unittest -v tests.unit.testSimpleTabular
```

`tests/unit` contiene las pruebas unitarias y `tests/integration`, las del ciclo de vida de las
fuentes a través de la API. No requieren PostgreSQL, Qdrant ni Ollama: usan SQLite y Qdrant en
memoria y simulan las respuestas de Ollama.
