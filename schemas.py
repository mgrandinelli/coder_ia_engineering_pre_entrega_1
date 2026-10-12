"""Esquemas Pydantic: mensajes, configuración del modelo, respuesta y settings."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

Provider = Literal["openai", "anthropic"]
ErrorKind = Literal[
    "rate_limit", "network", "timeout", "auth", "bad_request", "server", "refusal", "unknown"
]


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, igual para todos los proveedores."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros de generación, validados antes de salir a la red."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None  # None = modelo por defecto del proveedor
    temperature: float | None = Field(default=None, ge=0, le=2)  # None = la del proveedor
    max_tokens: int = Field(default=4096, gt=0, le=128_000)
    timeout_s: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)  # reintentos del SDK ante 429/5xx/red


class LLMErrorInfo(BaseModel):
    """Error controlado: lo que se devuelve en lugar de dejar escapar la excepción."""

    kind: ErrorKind
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    """Respuesta unificada. Si `error` no es None, `content` puede venir vacío."""

    content: str = ""
    provider: Provider
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    error: LLMErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Settings(BaseModel):
    """Configuración leída del entorno. Las claves son SecretStr: no salen en logs ni repr."""

    provider: Provider = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str | None = None
    anthropic_model: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        valores = {
            "provider": os.getenv("LLM_PROVIDER"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
        }
        # Una variable vacía en el .env cuenta como no definida.
        return cls(**{k: v.strip() for k, v in valores.items() if v and v.strip()})
