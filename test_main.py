"""Tests de la pre-entrega 1. No usan la red ni API keys: los SDK se reemplazan por dobles.

Ejecutar con: pytest
"""

from types import SimpleNamespace as NS

import anthropic
import openai
import pytest
from pydantic import ValidationError

import main
from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, ModelResponse, Settings

pytestmark = pytest.mark.asyncio

FRAGMENTOS = ["La ", "entropía ", "mide ", "el desorden."]
TEXTO = "".join(FRAGMENTOS)


def error_sdk(cls, mensaje="fallo simulado"):
    """Instancia una excepción del SDK sin armar una respuesta HTTP real."""
    exc = cls.__new__(cls)
    exc.args = (mensaje,)
    return exc


async def _iterar(items):
    for item in items:
        yield item


class OpenAIFalso:
    """Imita la parte de AsyncOpenAI que usa OpenAIClient."""

    def __init__(self, error=None, error_en_stream=None):
        self.error, self.error_en_stream = error, error_en_stream
        self.llamadas = []
        self.chat = NS(completions=NS(create=self._create))

    def with_options(self, **opciones):
        self.opciones = opciones
        return self

    async def _create(self, **params):
        self.llamadas.append(params)
        if self.error:
            raise self.error
        if params.get("stream"):
            return self._chunks()
        return NS(
            model=params["model"],
            choices=[NS(message=NS(content=TEXTO), finish_reason="stop")],
            usage=NS(prompt_tokens=12, completion_tokens=7),
        )

    async def _chunks(self):
        for f in FRAGMENTOS:
            yield NS(choices=[NS(delta=NS(content=f))])
            if self.error_en_stream:
                raise self.error_en_stream
        yield NS(choices=[])  # chunk final de uso: sin choices


class AnthropicFalso:
    """Imita la parte de AsyncAnthropic que usa AnthropicClient."""

    def __init__(self, error=None, stop_reason="end_turn"):
        self.error, self.stop_reason = error, stop_reason
        self.llamadas = []
        self.messages = NS(create=self._create, stream=self._stream)

    def with_options(self, **opciones):
        return self

    def _mensaje(self, params):
        return NS(
            model=params["model"],
            content=[NS(type="thinking", thinking=""), NS(type="text", text=TEXTO)],
            stop_reason=self.stop_reason,
            usage=NS(input_tokens=15, output_tokens=9),
        )

    async def _create(self, **params):
        self.llamadas.append(params)
        if self.error:
            raise self.error
        return self._mensaje(params)

    def _stream(self, **params):
        self.llamadas.append(params)
        falso = self

        class Flujo:
            async def __aenter__(self):
                if falso.error:
                    raise falso.error
                self.text_stream = _iterar(FRAGMENTOS)
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return falso._mensaje(params)

        return Flujo()


CASOS = [(OpenAIClient, OpenAIFalso, openai), (AnthropicClient, AnthropicFalso, anthropic)]
CONVERSACION = [
    ChatMessage(role="system", content="Sé breve."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]


# --- Abstracción y gestión de proveedores ---


@pytest.mark.parametrize("proveedor, clase", [("openai", OpenAIClient), ("anthropic", AnthropicClient)])
async def test_manager_carga_el_proveedor_segun_la_configuracion(monkeypatch, proveedor, clase):
    monkeypatch.setenv("LLM_PROVIDER", proveedor)

    manager = AsyncLLMManager()

    assert isinstance(manager.client, clase)
    assert isinstance(manager.client, BaseLLMClient)
    assert manager.provider == proveedor


async def test_manager_rechaza_un_proveedor_desconocido():
    with pytest.raises(ValidationError):
        AsyncLLMManager("gemini")


async def test_base_es_abstracta():
    with pytest.raises(TypeError):
        BaseLLMClient()


@pytest.mark.parametrize("clase, falso, _", CASOS)
async def test_generate_devuelve_la_misma_estructura_en_ambos_proveedores(clase, falso, _):
    manager = AsyncLLMManager(client=clase(sdk_client=falso()))

    respuesta = await manager.generate(CONVERSACION)

    assert isinstance(respuesta, ModelResponse)
    assert respuesta.ok
    assert respuesta.content == TEXTO
    assert respuesta.input_tokens and respuesta.output_tokens


async def test_anthropic_separa_el_system_y_envia_temperature():
    sdk = AnthropicFalso()

    await AnthropicClient(sdk_client=sdk).generate(CONVERSACION, ModelConfig(temperature=0.5))

    params = sdk.llamadas[0]
    assert params["system"] == "Sé breve."
    assert params["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert params["temperature"] == 0.5
    assert params["model"] == "claude-haiku-4-5"


async def test_sin_temperature_no_se_envia_el_parametro():
    sdk_openai, sdk_anthropic = OpenAIFalso(), AnthropicFalso()

    await OpenAIClient(sdk_client=sdk_openai).generate(CONVERSACION)
    await AnthropicClient(sdk_client=sdk_anthropic).generate(CONVERSACION)

    assert "temperature" not in sdk_openai.llamadas[0]
    assert "temperature" not in sdk_anthropic.llamadas[0]


async def test_openai_recibe_los_parametros_validados():
    sdk = OpenAIFalso()
    config = ModelConfig(temperature=0.5, max_tokens=100, timeout_s=5, max_retries=1)

    await OpenAIClient(sdk_client=sdk).generate(CONVERSACION, config)

    params = sdk.llamadas[0]
    assert params["messages"][0] == {"role": "system", "content": "Sé breve."}
    assert params["temperature"] == 0.5
    assert params["max_completion_tokens"] == 100
    assert sdk.opciones == {"timeout": 5, "max_retries": 1}


# --- Streaming asíncrono ---


@pytest.mark.parametrize("clase, falso, _", CASOS)
async def test_stream_entrega_los_fragmentos_de_a_uno(clase, falso, _):
    manager = AsyncLLMManager(client=clase(sdk_client=falso()))

    flujo = manager.stream("¿Qué es la entropía?")
    recibidos = [fragmento async for fragmento in flujo]

    assert hasattr(flujo, "__anext__")  # es un generador asíncrono
    assert recibidos == FRAGMENTOS
    assert manager.last_error is None


# --- Validación con Pydantic ---


@pytest.mark.parametrize(
    "campos",
    [{"temperature": -0.1}, {"temperature": 2.1}, {"max_tokens": 0}, {"max_retries": -1}, {"otro": 1}],
)
async def test_model_config_rechaza_valores_fuera_de_rango(campos):
    with pytest.raises(ValidationError):
        ModelConfig(**campos)


async def test_model_config_acepta_los_limites_de_temperature():
    assert ModelConfig(temperature=0).temperature == 0
    assert ModelConfig(temperature=2).temperature == 2


@pytest.mark.parametrize("mensaje", [{"role": "robot", "content": "hola"}, {"role": "user", "content": ""}])
async def test_chat_message_rechaza_rol_o_contenido_invalidos(mensaje):
    with pytest.raises(ValidationError):
        ChatMessage(**mensaje)


async def test_la_api_key_no_aparece_en_el_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secreto-123")

    settings = Settings.from_env()

    assert "sk-secreto-123" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-secreto-123"


# --- Resiliencia y manejo de errores ---


@pytest.mark.parametrize("clase, falso, sdk", CASOS)
@pytest.mark.parametrize(
    "excepcion, tipo, reintentable",
    [
        ("RateLimitError", "rate_limit", True),
        ("APIConnectionError", "network", True),
        ("APITimeoutError", "timeout", True),
        ("AuthenticationError", "auth", False),
    ],
)
async def test_generate_devuelve_un_error_controlado(clase, falso, sdk, excepcion, tipo, reintentable):
    cliente = clase(sdk_client=falso(error=error_sdk(getattr(sdk, excepcion))))

    respuesta = await cliente.generate()  # no lanza

    assert not respuesta.ok
    assert respuesta.error.kind == tipo
    assert respuesta.error.retryable is reintentable
    assert respuesta.content == ""


@pytest.mark.parametrize("clase, falso, sdk", CASOS)
async def test_stream_no_lanza_ante_un_rate_limit(clase, falso, sdk):
    cliente = clase(sdk_client=falso(error=error_sdk(sdk.RateLimitError)))

    recibidos = [f async for f in cliente.stream()]

    assert len(recibidos) == 1
    assert "[ERROR rate_limit]" in recibidos[0]
    assert cliente.last_error.kind == "rate_limit"


async def test_stream_conserva_lo_recibido_si_la_red_se_corta_a_mitad():
    corte = error_sdk(openai.APIConnectionError)
    cliente = OpenAIClient(sdk_client=OpenAIFalso(error_en_stream=corte))

    recibidos = [f async for f in cliente.stream()]

    assert recibidos[0] == FRAGMENTOS[0]
    assert "[ERROR network]" in recibidos[-1]


async def test_anthropic_informa_un_rechazo_como_error_controlado():
    cliente = AnthropicClient(sdk_client=AnthropicFalso(stop_reason="refusal"))

    respuesta = await cliente.generate()

    assert respuesta.error.kind == "refusal"


@pytest.mark.parametrize("proveedor", ["openai", "anthropic"])
async def test_sin_api_key_no_hay_crash(monkeypatch, proveedor):
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    manager = AsyncLLMManager(proveedor)

    respuesta = await manager.generate(config=ModelConfig(max_retries=0, timeout_s=5))
    recibidos = [f async for f in manager.stream(config=ModelConfig(max_retries=0, timeout_s=5))]

    assert not respuesta.ok
    assert "[ERROR" in recibidos[-1]


async def test_main_corre_sin_argumentos_y_sin_claves(monkeypatch, capsys):
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(variable, raising=False)

    await main.main()

    assert "El programa terminó sin interrumpirse." in capsys.readouterr().out
