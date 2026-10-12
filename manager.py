"""AsyncLLMManager: elige el proveedor según la configuración (patrón Factory)."""

from collections.abc import AsyncIterator

from clients import PREGUNTA_POR_DEFECTO, AnthropicClient, BaseLLMClient, Messages, OpenAIClient
from schemas import LLMErrorInfo, ModelConfig, ModelResponse, Provider, Settings

CLIENTES: dict[str, type[BaseLLMClient]] = {
    OpenAIClient.provider: OpenAIClient,
    AnthropicClient.provider: AnthropicClient,
}


class AsyncLLMManager:
    """Punto de entrada único: el resto de la aplicación no sabe qué proveedor hay detrás.

    El proveedor sale de `provider` o, si no se pasa, de la variable LLM_PROVIDER.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        settings: Settings | None = None,
        client: BaseLLMClient | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        if provider is not None:
            # Revalida: un proveedor desconocido falla acá, no en la primera llamada.
            settings = Settings.model_validate({**settings.model_dump(), "provider": provider})
        self.settings = settings
        self.client = client or CLIENTES[settings.provider](
            api_key=getattr(settings, f"{settings.provider}_api_key"),
            model=getattr(settings, f"{settings.provider}_model"),
        )

    @property
    def provider(self) -> str:
        return self.client.provider

    @property
    def last_error(self) -> LLMErrorInfo | None:
        return self.client.last_error

    async def generate(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> ModelResponse:
        return await self.client.generate(messages, config)

    async def stream(
        self, messages: Messages = PREGUNTA_POR_DEFECTO, config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        async for fragmento in self.client.stream(messages, config):
            yield fragmento
