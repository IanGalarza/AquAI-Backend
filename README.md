# AquAI Backend - Documentación Técnica

## Descripción General

AquAI es un proyecto para trabajar con fuentes de datos y conversaciones asistidas
por inteligencia artificial. Este repositorio contiene su backend en Python,
organizado en capas.

Los modelos del proyecto representan:

- Archivos originales y sus estados de procesamiento.
- Conjuntos de datos asociados a cada archivo.
- Conversaciones, mensajes y referencias a las fuentes utilizadas.

La persistencia está implementada con **SQLAlchemy**, **PostgreSQL** y **psycopg**,
siguiendo un enfoque **code-first**.

## Estructura del Proyecto

```text
src/aquai/
├── bootstrap.py                 # Inicialización de la aplicación
├── domain/
│   └── models/                  # Entidades de negocio
├── application/                 # Casos de uso, servicios e interfaces
├── infrastructure/
│   ├── config/
│   │   └── database.py          # Configuración de la conexión
│   ├── persistence/
│   │   ├── db_context.py        # Motor y sesiones de base de datos
│   │   ├── mapping.py           # Registro de los mapeos
│   │   └── configurations/      # Configuración de tablas y relaciones
│   └── providers/
│       └── database.py          # Proveedores de contexto y sesiones
└── presentation/                # Controllers y schemas

tests/
└── unit/                        # Pruebas de configuración y persistencia

docker-compose.yml              # Servicio de PostgreSQL
pyproject.toml                  # Dependencias y configuración del paquete
.env.example                    # Ejemplo de variables de entorno
```

## Arquitectura

El proyecto sigue una separación inspirada en **Clean Architecture**:

- **Domain**: define las entidades y sus relaciones, sin depender del ORM.
- **Application**: agrupa los casos de uso, servicios e interfaces.
- **Infrastructure**: implementa la configuración y el acceso a PostgreSQL.
- **Presentation**: organiza los controllers y schemas de entrada y salida.

## Modelos de Datos

Los modelos se encuentran en `src/aquai/domain/models`:

| Modelo | Responsabilidad |
| --- | --- |
| `Source` | Archivo original, contenido y metadatos. |
| `ProcessingStatus` | Estado de procesamiento de una fuente. |
| `Dataset` | Conjunto de datos obtenido de una fuente. |
| `Conversation` | Conversación del historial del chat. |
| `Message` | Mensaje, contenido estructurado y estado. |
| `MessageStatus` | Estado de un mensaje. |
| `MessageSource` | Referencia de un mensaje a una fuente y un dataset opcional. |

## Persistencia

- **`DatabaseSettings`** (`config/database.py`): carga y valida `DATABASE_URL`
  desde el entorno o el archivo `.env`.
- **`DbContext`** (`persistence/db_context.py`): administra el motor y las
  sesiones, con commit explícito desde el caso de uso, rollback y cierre de recursos.
- **Configuraciones** (`persistence/configurations`): definen los nombres SQL,
  tipos de columnas, claves y relaciones de cada modelo.
- **Providers** (`providers/database.py`): exponen `get_database_settings()`,
  `get_db_context()` y `get_db_session()` para obtener la configuración y las sesiones.

`Message.content_data` comienza con un documento vacío (`{}`). El contenido
binario de `Source` tiene carga diferida: se recupera al acceder a `file_content`
dentro de una sesión, o al incluirlo en la consulta con `undefer`.

### Inicialización

Registrá los mapeos una vez al arrancar la aplicación, antes de crear las
entidades que se van a persistir:

```python
from aquai.bootstrap import initialize_application

initialize_application()
```

`DbContext` administra las sesiones por separado. Para guardar cambios, el caso
de uso llama a `session.commit()` dentro del bloque `context.session()`.

### Catálogos de Estados

Los datos iniciales están definidos en `persistence/configurations/status_seeds.py`:

- **ProcessingStatus**: `uploaded`, `processing`, `ready`, `ready_with_warnings`,
  `review_required` y `failed` (IDs del 1 al 6, en ese orden).
- **MessageStatus**: `pending`, `processing`, `completed`, `failed` y `cancelled`
  (IDs del 1 al 5, en ese orden).

## Configuración y Uso

### Requisitos Previos

- Python **3.11 o superior**.
- Docker con **Docker Compose**.

Ejecutá los comandos desde la raíz del proyecto, en PowerShell.

### 1. Instalar Dependencias

Creá un entorno virtual e instalá el proyecto en modo editable:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. Configurar Variables de Entorno

Para la primera configuración, copiá el archivo de ejemplo:

```powershell
Copy-Item .env.example .env
```

Editá `.env` con los datos de tu base:

```dotenv
POSTGRES_DB=aquai
POSTGRES_USER=aquai_user
POSTGRES_PASSWORD=tu_password
POSTGRES_PORT=5432

DATABASE_URL=postgresql+psycopg://aquai_user:tu_password@localhost:5432/aquai
```

Las variables `POSTGRES_*` configuran el contenedor y `DATABASE_URL` configura
la conexión del backend. Usá los mismos datos en ambos. Si la contraseña tiene
caracteres especiales, codificalos en la URL; por ejemplo, `@` como `%40`.

### 3. Levantar PostgreSQL

```powershell
docker compose up -d postgres
```

Para consultar el estado del servicio:

```powershell
docker compose ps
```

## Pruebas

Ejecutá las pruebas con:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

Las pruebas cubren la validación de la configuración, los mapeos de PostgreSQL,
las relaciones entre modelos y el manejo de las sesiones.
