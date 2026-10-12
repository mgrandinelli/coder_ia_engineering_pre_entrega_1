"""Pre-entrega 1 - Script de validación del Unified Async LLM Client.

Uso: python main.py [openai|anthropic]   (sin argumento usa LLM_PROVIDER del .env)
"""

import asyncio
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from clients import PREGUNTA_POR_DEFECTO, AnthropicClient, BaseLLMClient, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, ModelResponse

load_dotenv()

__all__ = [
    "AnthropicClient",
    "AsyncLLMManager",
    "BaseLLMClient",
    "ChatMessage",
    "ModelConfig",
    "ModelResponse",
    "OpenAIClient",
    "main",
]


async def main(provider: str | None = None) -> None:
    try:
        manager = AsyncLLMManager(provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        return

    mensajes = [
        ChatMessage(role="system", content="Respondé en español, en dos o tres oraciones."),
        ChatMessage(role="user", content=PREGUNTA_POR_DEFECTO),
    ]
    print(f"Proveedor: {manager.provider} | modelo: {manager.client.model}")
    print(f"Pregunta: {PREGUNTA_POR_DEFECTO}")

    print("\n=== Modo normal ===")
    respuesta = await manager.generate(mensajes)
    if respuesta.ok:
        print(respuesta.content)
        print(f"(tokens: {respuesta.input_tokens} entrada, {respuesta.output_tokens} salida)")
    else:
        print(f"[ERROR {respuesta.error.kind}] {respuesta.error.message}")

    print("\n=== Modo streaming ===")
    async for fragmento in manager.stream(mensajes):
        print(fragmento, end="", flush=True)
    print()

    print("\n=== Validación con Pydantic ===")
    try:
        ModelConfig(temperature=3.5)
    except ValidationError as exc:
        print(f"temperature=3.5 rechazada: {exc.errors()[0]['msg']}")

    print("\nEl programa terminó sin interrumpirse.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
