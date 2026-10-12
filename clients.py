"""Clientes asíncronos de LLM bajo una interfaz común (OpenAI y Anthropic)."""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic
import openai
from pydantic import SecretStr, TypeAdapter

from schemas import ChatMessage, LLMErrorInfo, ModelConfig, ModelResponse, Provider

PREGUNTA_POR_DEFECTO = "¿Qué es la entropía?"

Messages = str | list[ChatMessage] | list[dict[str, str]]
_MENSAJES = TypeAdapter(list[ChatMessage])


class BaseLLMClient(ABC):
    """Interfaz común. `generate` y `stream` nunca lanzan errores de la API:
    los devuelven como `LLMErrorInfo` (en `ModelResponse.error` o en `last_error`)."""

    provider: ClassVar[Provider]
    default_model: ClassVar[str]
    api_key_env: ClassVar[str]
    _sdk_module: ClassVar[Any]  # openai o anthropic: comparten nombres de excepciones

    def __init__(
        self,
        api_key: SecretStr | str | None = None,
        model: str | None = None,
        sdk_client: Any = None,  # para inyectar un doble en los tests
    ) -> None:
        clave = api_key or os.getenv(self.api_key_env) or None
        self._api_key = SecretStr(clave) if isinstance(clave, str) else clave
        self.model = model or self.default_model
        self._sdk_client = sdk_client
        self.last_error: LLMErrorInfo | None = None

    @abstractmethod
    async def generate(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> ModelResponse:
        """Devuelve la respuesta completa."""

    @abstractmethod
    def stream(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto. Si la API falla, el último
        fragmento es `[ERROR tipo] mensaje` y el detalle queda en `last_error`."""

    @abstractmethod
    def _crear_sdk(self) -> Any:
        """Construye el cliente asíncrono del SDK oficial."""

    def _sdk(self, config: ModelConfig) -> Any:
        if self._sdk_client is None:
            self._sdk_client = self._crear_sdk()
        return self._sdk_client.with_options(
            timeout=config.timeout_s, max_retries=config.max_retries
        )

    @staticmethod
    def _normalizar(messages: Messages) -> list[ChatMessage]:
        if isinstance(messages, str):
            return [ChatMessage(role="user", content=messages)]
        return _MENSAJES.validate_python(messages)

    def _traducir_error(self, exc: Exception) -> LLMErrorInfo:
        """Convierte la excepción del SDK en un error controlado y lo guarda en `last_error`."""
        sdk = self._sdk_module
        mensaje = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        # De más específica a más general: APITimeoutError hereda de APIConnectionError.
        if isinstance(exc, sdk.APITimeoutError):
            info = LLMErrorInfo(kind="timeout", message=mensaje, retryable=True)
        elif isinstance(exc, sdk.APIConnectionError):
            info = LLMErrorInfo(kind="network", message=mensaje, retryable=True)
        elif isinstance(exc, sdk.RateLimitError):
            info = LLMErrorInfo(kind="rate_limit", message=mensaje, retryable=True)
        elif isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            info = LLMErrorInfo(kind="auth", message=mensaje)
        elif isinstance(exc, sdk.APIStatusError):
            if exc.status_code >= 500:
                info = LLMErrorInfo(kind="server", message=mensaje, retryable=True)
            else:
                info = LLMErrorInfo(kind="bad_request", message=mensaje)
        elif isinstance(exc, _CredencialFaltante):
            info = LLMErrorInfo(kind="auth", message=mensaje)
        else:
            info = LLMErrorInfo(kind="unknown", message=mensaje)
        self.last_error = info
        return info

    def _respuesta_error(self, info: LLMErrorInfo, model: str) -> ModelResponse:
        self.last_error = info
        return ModelResponse(provider=self.provider, model=model, error=info)

    @staticmethod
    def _fragmento_error(info: LLMErrorInfo) -> str:
        return f"\n[ERROR {info.kind}] {info.message}"


class _CredencialFaltante(Exception):
    pass


class OpenAIClient(BaseLLMClient):
    provider = "openai"
    default_model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"
    _sdk_module = openai

    def _crear_sdk(self) -> openai.AsyncOpenAI:
        if self._api_key is None:
            raise _CredencialFaltante("Falta OPENAI_API_KEY (revisá tu archivo .env)")
        return openai.AsyncOpenAI(api_key=self._api_key.get_secret_value())

    def _params(self, messages: Messages, config: ModelConfig) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": config.model or self.model,
            "messages": [m.model_dump() for m in self._normalizar(messages)],
            "max_completion_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            params["temperature"] = config.temperature
        return params

    async def generate(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> ModelResponse:
        config = config or ModelConfig()
        self.last_error = None
        model = config.model or self.model
        try:
            respuesta = await self._sdk(config).chat.completions.create(
                **self._params(messages, config)
            )
        except Exception as exc:
            return self._respuesta_error(self._traducir_error(exc), model)

        eleccion = respuesta.choices[0]
        uso = respuesta.usage
        return ModelResponse(
            content=eleccion.message.content or "",
            provider=self.provider,
            model=respuesta.model,
            input_tokens=uso.prompt_tokens if uso else None,
            output_tokens=uso.completion_tokens if uso else None,
            finish_reason=eleccion.finish_reason,
        )

    async def stream(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        config = config or ModelConfig()
        self.last_error = None
        try:
            flujo = await self._sdk(config).chat.completions.create(
                **self._params(messages, config), stream=True
            )
            async for chunk in flujo:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            yield self._fragmento_error(self._traducir_error(exc))


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    default_model = "claude-opus-5-5"
    api_key_env = "ANTHROPIC_API_KEY"
    _sdk_module = anthropic

    def _crear_sdk(self) -> anthropic.AsyncAnthropic:
        # Sin clave explícita, el SDK resuelve las credenciales del entorno por su cuenta.
        if self._api_key is None:
            return anthropic.AsyncAnthropic()
        return anthropic.AsyncAnthropic(api_key=self._api_key.get_secret_value())

    def _params(self, messages: Messages, config: ModelConfig) -> dict[str, Any]:
        normalizados = self._normalizar(messages)
        params: dict[str, Any] = {
            "model": config.model or self.model,
            "max_tokens": config.max_tokens,
            # Anthropic no acepta el rol "system" al inicio de `messages`: va aparte.
            "messages": [m.model_dump() for m in normalizados if m.role != "system"],
        }
        sistema = "\n\n".join(m.content for m in normalizados if m.role == "system")
        if sistema:
            params["system"] = sistema
        # `temperature` no se envía: los modelos Claude actuales la rechazan con un 400.
        return params

    _RECHAZO = LLMErrorInfo(kind="refusal", message="El modelo rechazó la solicitud")

    async def generate(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> ModelResponse:
        config = config or ModelConfig()
        self.last_error = None
        model = config.model or self.model
        try:
            respuesta = await self._sdk(config).messages.create(**self._params(messages, config))
        except Exception as exc:
            return self._respuesta_error(self._traducir_error(exc), model)

        if respuesta.stop_reason == "refusal":
            return self._respuesta_error(self._RECHAZO, respuesta.model)
        return ModelResponse(
            # `content` es una lista de bloques (texto, thinking, ...): solo interesa el texto.
            content="".join(b.text for b in respuesta.content if b.type == "text"),
            provider=self.provider,
            model=respuesta.model,
            input_tokens=respuesta.usage.input_tokens,
            output_tokens=respuesta.usage.output_tokens,
            finish_reason=respuesta.stop_reason,
        )

    async def stream(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        config = config or ModelConfig()
        self.last_error = None
        try:
            async with self._sdk(config).messages.stream(
                **self._params(messages, config)
            ) as flujo:
                async for texto in flujo.text_stream:
                    yield texto
                final = await flujo.get_final_message()
            if final.stop_reason == "refusal":
                self.last_error = self._RECHAZO
                yield self._fragmento_error(self._RECHAZO)
        except Exception as exc:
            yield self._fragmento_error(self._traducir_error(exc))
