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
│   ├── migrations/
│   │   ├── env.py               # Integración de Alembic con la configuración
│   │   └── versions/            # Historial de migraciones
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
alembic.ini                     # Configuración de Alembic
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

## Migraciones

Alembic utiliza la conexión de `DatabaseSettings` y la metadata de
`infrastructure/persistence/mapping.py`. Los mapeos se registran antes de
comparar el esquema con los modelos.

La migración **InitialCreate** crea las siete tablas del proyecto e incluye los
catálogos iniciales de `ProcessingStatus` y `MessageStatus`. Revisá el archivo en
`src/aquai/infrastructure/migrations/versions` antes de aplicarlo.

Desde la raíz del proyecto, en PowerShell:

| Operación | Comando |
| --- | --- |
| Crear una migración desde los modelos | `.\.venv\Scripts\python.exe -m alembic revision --autogenerate -m "NombreMigracion"` |
| Aplicar todas las migraciones pendientes | `.\.venv\Scripts\python.exe -m alembic upgrade head` |
| Revertir la última migración aplicada | `.\.venv\Scripts\python.exe -m alembic downgrade -1` |
| Consultar la versión actual de la base | `.\.venv\Scripts\python.exe -m alembic current --verbose` |
| Ver el historial | `.\.venv\Scripts\python.exe -m alembic history --verbose` |

Para generar el SQL y revisarlo sin aplicarlo:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head --sql
```

Al generar otra migración con `--autogenerate`, la base debe estar al día con el
historial existente. La revisión de columnas, restricciones y datos iniciales
forma parte del proceso antes de aplicar cada archivo.

## Pruebas

Ejecutá las pruebas con:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

Las pruebas cubren la validación de la configuración, los mapeos de PostgreSQL,
las relaciones entre modelos y el manejo de las sesiones.

## Procesamiento Tabular Experimental

El endpoint `POST /experimental/tabular/evaluate` recibe un CSV o XLSX y permite
evaluar recetas de preparación generadas por distintos modelos locales. Este flujo
no necesita PostgreSQL ni Qdrant y no persiste archivos, entidades ni recetas.
El archivo original se mantiene disponible durante la solicitud y se libera al
terminar. La carga HTTP puede usar almacenamiento temporal administrado por FastAPI.

### Iniciar y Evaluar

Iniciá Ollama y asegurate de tener disponible el modelo que quieras probar.
La URL del proveedor y el modelo predeterminado se configuran con
`TABULAR_OLLAMA_BASE_URL` y `TABULAR_DEFAULT_MODEL` en `.env` o en el entorno.
El modelo predeterminado es `qwen3.5:9b`, con contexto de 16384 tokens y
`think=false`. El contexto y el pensamiento se configuran con
`TABULAR_LLM_NUM_CTX` y `TABULAR_LLM_THINK`.
El tiempo máximo de respuesta es 600 segundos, configurable mediante
`TABULAR_LLM_TIMEOUT_SECONDS`; no hay reintentos ni cambios automáticos de modelo,
contexto o pensamiento.
El proyecto no descarga modelos automáticamente.

```powershell
.\.venv\Scripts\python.exe -m uvicorn aquai.main:app --reload
```

La documentación interactiva está disponible en `http://localhost:8000/docs`.
El campo multipart `file` es obligatorio. El parámetro de consulta `model`
reemplaza el modelo configurado para esa evaluación. El campo multipart opcional
`known_rules` permite agregar reglas de preparación (hasta 4000 caracteres).
El campo multipart `rule_profile` selecciona explícitamente un perfil:
`biological_counts`, `self_sampling`, `belen_cyanobacteria` o `cyanobacterial_alerts`.
Sin perfil se usan únicamente las reglas comunes y las adicionales.
Un perfil desconocido devuelve HTTP 400 antes de llamar al modelo.

```powershell
curl.exe -X POST "http://localhost:8000/experimental/tabular/evaluate" -F "file=@D:/datos/recuentos.xlsx"
curl.exe -X POST "http://localhost:8000/experimental/tabular/evaluate?model=otro-modelo:tag" -F "file=@D:/datos/recuentos.xlsx" -F "known_rules=Conservar todos los registros."
```

Para comparar modelos, enviá el mismo archivo y las mismas reglas en cada
solicitud y conservá las respuestas del endpoint.

### Flujo y Contrato

1. Application valida nombre, extensión, contenido y tamaño del archivo.
2. Infrastructure inspecciona el archivo con Pandas y OpenPyXL, sin una LLM.
3. Application compone las reglas comunes, el perfil seleccionado y las reglas
   adicionales. Su versión y procedencia se conservan en `applied_rules`.
   Ollama recibe un resumen estructural con muestras acotadas y el esquema JSON
   de la receta. Se utiliza la API de chat con salida estructurada y temperatura 0.
4. Application valida la receta con Pydantic. Tolera texto o bloques Markdown
   alrededor del objeto JSON y conserva la respuesta original sin modificarla.
5. Infrastructure ejecuta sólo las operaciones implementadas explícitamente.
6. Application clasifica el resultado a partir de las comprobaciones determinísticas.

Las filas de la receta empiezan en **1**, incluyendo los registros del CSV.
Un valor entrecomillado con saltos de línea pertenece al mismo registro.
Un CSV se representa como una hoja llamada `data`. Los encabezados no se eliminan
durante la inspección. `header_rows` puede tener varias filas; sus valores se
combinan con espacios. Sin encabezados se generan nombres `column_1`, `column_2`,
etc. `data_start_row` y `data_end_row` definen el rango de datos original e inclusivo.
Cada entrada de `sheets` representa un bloque. Una hoja puede aparecer varias
veces con `block_id` distintos. `data_start_column` y `data_end_column` delimitan
columnas originales, también desde 1 e inclusivas. `role` distingue
`observations`, `total`, `legend`, `derived`, `summary`, `header` y `unknown`.
Los nueve campos de cada bloque son obligatorios; los cuatro límites deben ser
enteros positivos explícitos. `block_id` debe ser no vacío y único en toda la
receta; `role` no tiene un valor predeterminado. Las respuestas antiguas con
campos omitidos se rechazan antes de ejecutar, sin completar decisiones.
Los bloques conservan los datos de los totales y las leyendas en tablas separadas;
no implican descartarlos. Los rangos de datos no pueden solaparse, aunque varios
bloques pueden compartir sus filas de encabezado.
`ready` requiere bloques definidos, sin `unknown`, advertencias ni solicitud de
revisión. `review_required` exige `requires_human_review=true` y una explicación.
Puede conservar bloques `unknown` sin transformaciones o una propuesta parcial
o vacía si no se pueden justificar los límites. Esa revisión no acredita una
preparación completa. La cobertura y el formato válido tampoco acreditan por sí
solos que los roles y las relaciones de los datos sean correctos.

```json
{
  "status": "ready",
  "sheets": [
    {
      "sheet_name": "data",
      "block_id": "sample_counts",
      "role": "observations",
      "header_rows": [1],
      "data_start_row": 2,
      "data_end_row": 20,
      "data_start_column": 1,
      "data_end_column": 2,
      "transformations": [
        {"operation": "strip_strings"},
        {"operation": "convert_type", "columns": ["count"], "target_type": "integer"}
      ]
    }
  ],
  "warnings": [],
  "requires_human_review": false
}
```

| Operación | Parámetros | Comportamiento |
| --- | --- | --- |
| `drop_empty_rows` | Ninguno | Elimina filas completamente vacías. |
| `drop_empty_columns` | Ninguno | Elimina columnas completamente vacías. |
| `strip_strings` | `columns` opcional | Quita espacios al principio y al final de cadenas. |
| `rename_columns` | `mapping` | Renombra columnas existentes. |
| `convert_type` | `columns`, `target_type` | Convierte a `number`, `integer`, `datetime` o `boolean`; conserva el original cuando falla y registra sus coordenadas. |

Las columnas de cada operación se referencian por su nombre vigente después de
las operaciones anteriores. Campos extra, operaciones desconocidas o instrucciones
de ejecución de código se rechazan. Las fórmulas de Excel no se recalculan: se usan
los valores guardados en el archivo; las fórmulas sin valor guardado requieren revisión.
Se conservan las expresiones completas durante el procesamiento y se devuelven
muestras con coordenada, expresión, valor cacheado y disponibilidad. Las conversiones
que perderían precisión o ceros iniciales de códigos, y las fechas ambiguas o
períodos, requieren revisión y conservan el valor original.

Las reglas editables viven en `src/aquai/infrastructure/config/tabular_rules.json`.
Application las compone mediante un catálogo inyectado; sólo recibe las reglas
del perfil elegido. Las observaciones específicas son condicionales y se contrastan
con el archivo recibido. Los conflictos explícitos de reglas adicionales requieren
revisión; el modelo también debe declarar cualquier contradicción que identifique.
El prompt general está en `src/aquai/application/agents/prompts/tabular.py` y
el mensaje enviado a Ollama se construye en `src/aquai/infrastructure/providers/llm/ollama.py`.

### Resultado y Validación

La respuesta incluye `model`, `structural_summary`, `recipe`, `llm_response`,
`execution`, `validation`, `metrics` y `applied_rules`. `llm_response.content` conserva el texto
original del modelo y `metadata` incluye las métricas devueltas por Ollama, como
duraciones, conteos de tokens y razonamiento cuando el proveedor lo devuelve.
Las duraciones del pipeline se expresan en milisegundos; las duraciones originales
de Ollama conservan sus unidades del proveedor (nanosegundos).

El resumen incluye dimensiones físicas, muestras con números de fila, nombres
iniciales, nulos, tipos inferidos, hojas, celdas combinadas, fórmulas y candidatos
de encabezado. `column_indices` relaciona los metadatos y las muestras con las
posiciones originales de las columnas, que empiezan en 1. En tablas anchas se
seleccionan columnas representativas y se incluyen regiones con muestras de sus
límites, centro y final. Las dimensiones y los conteos totales se calculan sobre
toda la hoja. Los cambios de estructura, encabezados repetidos, totales y bandas
derivadas se detectan con heurísticas genéricas; no establecen por sí solos una
interpretación científica. Se omiten del mensaje metadatos con valores predeterminados
redundantes, conservando coordenadas, regiones y conteos.
Las fórmulas extensas se muestran al modelo como vistas previas explícitas
(`formula_is_preview` y longitud original); la inspección y la ejecución conservan
su expresión completa. Los textos acotados se informan con sus conteos y posiciones.

La ejecución devuelve dimensiones antes/después, columnas finales, tipos,
muestras procesadas, filas excluidas, filas vacías eliminadas, columnas vacías,
conversiones fallidas, fórmulas sin valor y errores por hoja. `rows_before` cuenta
las filas del rango de datos elegido, después de separar los encabezados.
`excluded_nonempty_rows` cuenta registros no vacíos excluidos por ese rango.
Las celdas de error de Excel también se informan y requieren revisión. Las
conversiones numéricas que perderían precisión se contabilizan como fallidas.
`execution.coverage` comprueba la unión de bloques y encabezados contra las
celdas originales no vacías, incluyendo fórmulas. La separación declarada no se
confunde con pérdida. Las celdas sin tratamiento y los roles estructurales mezclados
requieren revisión; los solapamientos invalidan la ejecución. `validation.findings`
identifica el código del hallazgo, hoja, bloque, región, celda o rango disponible.

La integración sigue el contrato oficial de [chat de Ollama](https://docs.ollama.com/api/chat)
y de [salida estructurada](https://docs.ollama.com/capabilities/structured-outputs).

| Estado | Significado |
| --- | --- |
| `ready` | La ejecución produjo datos y no se detectaron problemas. |
| `ready_with_warnings` | La ejecución produjo datos con advertencias o limpieza de elementos vacíos. |
| `review_required` | Se detectaron pérdidas, conversiones fallidas, hojas omitidas con contenido, fórmulas sin valor o incertidumbre declarada. |
| `failed` | La receta es inválida, falló el proveedor o no pudo obtenerse una estructura utilizable. |

Una evaluación completada devuelve HTTP 200 incluso cuando `validation.status`
es `failed`, para poder comparar fallos entre modelos. Una receta inválida conserva
la respuesta original, deja `recipe` en `null` y no ejecuta transformaciones.
Un fallo de conexión con Ollama devuelve `llm_response` en `null` y un error legible.
Los archivos vacíos, ilegibles o no admitidos devuelven HTTP 400, los que exceden
el tamaño configurado devuelven HTTP 413 y los parámetros HTTP inválidos devuelven 422.

### Límites

Los valores predeterminados son 10 MiB por archivo, 2 millones de celdas en total,
30 hojas y 100 MiB de contenido expandido para XLSX. También se limita la lectura
a 1000 columnas por hoja. Las muestras estructurales tienen hasta 12 filas y 16
columnas; las muestras de ejecución muestran hasta 50 columnas. Los textos se
acotan a 200 caracteres. Las listas extensas de posiciones vacías y fórmulas se
acotan, conservando conteos y advertencias para indicar la truncación.
Los límites configurables figuran en `.env.example`.

Esta versión no procesa PDF, no ejecuta código generado por el modelo y no
implementa interacción humana para resolver las revisiones.

### Pruebas del Pipeline

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

Las pruebas del pipeline usan archivos CSV/XLSX en memoria y un transporte HTTP
simulado para Ollama. Verifican inspección, ejecución, validación, errores de
entrada, contrato de recetas, intercambio de modelos y el endpoint completo sin
conectar con bases de datos. La calidad de las recetas de un modelo real se evalúa
enviando archivos al endpoint con Ollama en funcionamiento.
