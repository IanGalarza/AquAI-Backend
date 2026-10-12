# AquAI Backend

Backend de AquAI para cargar archivos CSV y XLSX, gestionar fuentes y almacenar
sus conjuntos de datos. Ofrece una API HTTP con FastAPI y capacidades semánticas
mediante modelos locales.

## Arquitectura

El proyecto sigue Clean Architecture y se organiza en cuatro capas:

| Capa | Responsabilidad |
|---|---|
| Domain | Entidades y estados del negocio. |
| Application | Casos de uso, contratos, interfaces y DTOs. |
| Infrastructure | Persistencia, migraciones e integración con servicios externos. |
| Presentation | Controladores HTTP y manejo de solicitudes y errores de la API. |

PostgreSQL almacena las fuentes y los datos. Ollama ejecuta los modelos de
lenguaje y embeddings. Qdrant almacena el índice semántico.

## Requisitos

- Python 3.11 o superior.
- Docker con Docker Compose.
- Ollama instalado en el equipo.

Los comandos siguientes están preparados para PowerShell y deben ejecutarse
desde la raíz del repositorio.

## Instalación

### 1. Crear el entorno virtual e instalar las dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 2. Configurar las variables de entorno

Crear `.env` a partir del ejemplo si todavía no existe:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item .env.example .env
}
```

Editar `.env` con las credenciales y conexiones del entorno. `DATABASE_URL` debe
coincidir con las variables `POSTGRES_*`. Los caracteres especiales del usuario
y la contraseña deben codificarse en la URL; por ejemplo, `@` como `%40`.

El archivo también permite configurar Ollama, Qdrant y los límites de carga.

### 3. Iniciar PostgreSQL y Qdrant

Con Docker iniciado:

```powershell
docker compose up -d postgres qdrant
```

### 4. Instalar los modelos de Ollama

Ollama debe estar ejecutándose. Si no está iniciado, ejecutar `ollama serve`
en otra terminal y dejarla abierta.

Luego, instalar los modelos definidos en `.env`. Con la configuración predeterminada:

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 5. Aplicar las migraciones

```powershell
python -m alembic upgrade head
```

## Ejecución

Con PostgreSQL, Qdrant y Ollama iniciados:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn aquai.main:app --reload
```

La API queda disponible en `http://localhost:8000`. Para consultar y probar los
endpoints, abrir [Swagger UI](http://localhost:8000/docs), seleccionar una
operación y usar **Try it out** y **Execute**.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/sources` | Carga un CSV o XLSX mediante `multipart/form-data`, con el campo `file`. Responde con HTTP 202 y el identificador de la fuente. |
| GET | `/sources` | Lista las fuentes con paginación y filtros. |
| GET | `/sources/{id}` | Consulta el estado de una fuente y sus datasets, columnas y conteos. |
| PATCH | `/sources/{id}` | Activa o desactiva una fuente. |

`GET /sources` acepta los parámetros opcionales `page` (predeterminado: 1),
`page_size` (predeterminado: 20; máximo: 100), `name` y `processing_status`.
El parámetro `{id}` de las rutas de detalle y activación es un UUID.

Para activar una fuente, enviar este cuerpo JSON a `PATCH /sources/{id}`:

```json
{
  "isActive": true
}
```

Usar `false` para desactivarla. Los esquemas de las solicitudes y respuestas
están disponibles en Swagger UI.
