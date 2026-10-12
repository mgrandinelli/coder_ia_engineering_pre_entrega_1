# Pre-entrega 1: Cliente de LLM robusto y asíncrono

Curso AI Engineering (Coderhouse), módulo 1.

Un "Unified Async LLM Client": una sola interfaz asíncrona para hablar con OpenAI o con Anthropic. El proveedor se elige con una variable de entorno, sin tocar el código que lo usa.

## Cómo ejecutarlo

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completá tu clave
python main.py
```

`main.py` hace la pregunta "¿Qué es la entropía?" en modo normal y en streaming con el proveedor configurado. Para probar el otro proveedor sin editar el `.env`:

```bash
python main.py anthropic
python main.py openai
```

## Variables de entorno

| Variable | Obligatoria | Qué es |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `openai`) | Proveedor a usar: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si usás OpenAI | Clave de la API de OpenAI |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | Clave de la API de Anthropic |
| `OPENAI_MODEL` | No (por defecto `gpt-4o-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (por defecto `claude-opus-5-5`) | Modelo de Anthropic |

Solo hace falta la clave del proveedor que vayas a usar. El archivo `.env` está en `.gitignore`; al repo solo se sube `.env.example`.

## Archivos

| Archivo | Contenido |
|---|---|
| `schemas.py` | Esquemas Pydantic: `ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMErrorInfo`, `Settings` |
| `clients.py` | `BaseLLMClient` (clase base abstracta), `OpenAIClient` y `AnthropicClient` |
| `manager.py` | `AsyncLLMManager`: carga el cliente según la configuración |
| `main.py` | Script de validación: modo normal y streaming |
| `test_main.py` | 34 tests, sin red ni claves |
| `.env.example` | Plantilla de variables de entorno |

## Cómo se usa desde código

```python
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig

manager = AsyncLLMManager()             # usa LLM_PROVIDER
manager = AsyncLLMManager("anthropic")  # o se elige explícitamente

mensajes = [
    ChatMessage(role="system", content="Respondé en español."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]

# Modo normal
respuesta = await manager.generate(mensajes, ModelConfig(max_tokens=500))
if respuesta.ok:
    print(respuesta.content)
else:
    print(respuesta.error.kind, respuesta.error.message)

# Streaming
async for fragmento in manager.stream(mensajes):
    print(fragmento, end="", flush=True)
```

## Cómo cubre cada criterio de la rúbrica

### Abstracción y gestión de proveedores

- `BaseLLMClient` es una clase abstracta con `generate()` y `stream()`. `OpenAIClient` y `AnthropicClient` heredan de ella y usan los SDK oficiales en su versión asíncrona (`AsyncOpenAI` y `AsyncAnthropic`).
- `AsyncLLMManager` aplica el patrón Factory: lee `LLM_PROVIDER` y crea el cliente que corresponde.
- Las diferencias entre proveedores quedan dentro de cada cliente. Dos ejemplos: OpenAI devuelve el texto en `choices[0].message.content` y Anthropic en una lista de bloques; OpenAI acepta el rol `system` dentro de los mensajes y Anthropic lo recibe en un parámetro aparte. El resto de la aplicación siempre recibe un `ModelResponse`.

### Streaming asíncrono

`stream()` es un generador asíncrono: recorre el stream del SDK con `async for` y entrega cada fragmento con `yield` apenas llega. En OpenAI usa `chat.completions.create(stream=True)`; en Anthropic, `messages.stream()` y su `text_stream`.

### Validación con Pydantic

- `ModelConfig` valida los parámetros antes de salir a la red: `temperature` entre 0 y 2, `max_tokens` mayor que 0, `timeout_s` mayor que 0, `max_retries` entre 0 y 10. No acepta campos desconocidos.
- `ChatMessage` solo acepta los roles `system`, `user` y `assistant`, y contenido no vacío.
- `Settings` guarda las claves como `SecretStr`, así no aparecen en logs ni en mensajes de error.

### Resiliencia y manejo de errores

Ninguna excepción de la API sale de `generate()` ni de `stream()`. Cada cliente la traduce a un `LLMErrorInfo` con tres datos: `kind`, `message` y `retryable`.

| `kind` | Cuándo | `retryable` |
|---|---|---|
| `rate_limit` | Límite de tasa o de cuota (HTTP 429) | Sí |
| `network` | Falla de conexión | Sí |
| `timeout` | La llamada superó `timeout_s` | Sí |
| `server` | Error del proveedor (HTTP 5xx) | Sí |
| `auth` | Clave inválida, faltante o sin permisos | No |
| `bad_request` | Solicitud inválida (otros HTTP 4xx) | No |
| `refusal` | El modelo rechazó la solicitud | No |
| `unknown` | Cualquier otro error | No |

- En `generate()` el error viene en `ModelResponse.error`, y `respuesta.ok` es `False`.
- En `stream()` el último fragmento es `[ERROR tipo] mensaje` y el detalle queda en `manager.last_error`. Si la red se corta a mitad de la respuesta, los fragmentos ya recibidos se conservan.
- Antes de devolver el error, los SDK reintentan solos los fallos transitorios (429, 5xx y red) con espera exponencial. `ModelConfig.max_retries` define cuántas veces (2 por defecto).

Las llamadas usan siempre `await` sobre los clientes asíncronos, así que el event loop nunca se bloquea.

## Salida esperada

Con una clave válida, `python main.py` imprime la respuesta completa, luego la misma pregunta en streaming (el texto aparece de a poco) y por último la validación de Pydantic.

Sin clave configurada, el programa no se rompe. Esta es la salida real:

```
Proveedor: openai | modelo: gpt-4o-mini
Pregunta: ¿Qué es la entropía?

=== Modo normal ===
[ERROR auth] Falta OPENAI_API_KEY (revisá tu archivo .env)

=== Modo streaming ===

[ERROR auth] Falta OPENAI_API_KEY (revisá tu archivo .env)

=== Validación con Pydantic ===
temperature=3.5 rechazada: Input should be less than or equal to 2

El programa terminó sin interrumpirse.
```

## Tests

```bash
pytest
```

Son 34 tests y tardan unos 2 segundos. No usan la red ni claves: reemplazan los SDK por dobles de prueba que se inyectan con el parámetro `sdk_client` de cada cliente. Cubren los cuatro criterios de la rúbrica:

- **Proveedores:** el manager carga el cliente correcto, rechaza un proveedor desconocido y ambos clientes devuelven la misma estructura.
- **Streaming:** los fragmentos llegan de a uno desde un generador asíncrono.
- **Validación:** valores fuera de rango, roles inválidos y que la clave no aparezca en el `repr`.
- **Errores:** límite de tasa, red, timeout y autenticación en ambos proveedores, corte de red a mitad del stream y ejecución sin claves.

## Decisiones de diseño

- **`temperature` no se envía a Anthropic.** `ModelConfig` la valida entre 0 y 2 como pide la consigna, pero los modelos Claude actuales rechazan ese parámetro con un error 400, así que `AnthropicClient` lo omite. En OpenAI solo se envía si se define.
- **`max_tokens` por defecto en 4096.** Es un valor que aceptan los modelos de ambos proveedores. En Claude ese límite incluye el razonamiento interno del modelo, por eso no conviene bajarlo demasiado.
- **`gather` y `TaskGroup`.** Los clientes son corrutinas comunes, así que se pueden lanzar varias llamadas a la vez con `asyncio.gather` o, para que una falla cancele a las demás, con `asyncio.TaskGroup`.
