from src.config import settings
from src.llm_factory import LLMFactory, LLMProvider

# Import all callers to register them with the factory
import gemini_caller
import qwen_caller


class LLMService:
    def __init__(self, provider: str = None, model: str = None):
        """
        Initialize LLM Service with fallback priority: Gemini first, then Qwen

        Args:
            provider: LLM provider (gemini, qwen). If None, uses fallback priority
            model: Model name. Defaults to provider's default model
        """
        self.provider_priority = ["gemini", "qwen"]  # Gemini first, then Qwen
        self.provider = provider
        self.model = model
        self.llm = None

        # Try to initialize LLM with fallback
        self._initialize_llm()

        self.system_prompt = "You are a helpful assistant plus a very high class problem solver, and you give high class digital supports. Answer questions using the provided context."

    def _initialize_llm(self):
        """Initialize LLM with fallback mechanism"""
        providers_to_try = [self.provider] if self.provider else self.provider_priority

        for provider in providers_to_try:
            try:
                if provider == "gemini":
                    api_key = settings.gemini_api_key
                    model = self.model or settings.gemini_default_model
                elif provider == "qwen":
                    api_key = settings.qwen_api_key
                    model = self.model or settings.qwen_default_model
                else:
                    continue

                self.llm = LLMFactory.create_caller(
                    provider=LLMProvider(provider),
                    api_key=api_key,
                    model=model,
                    temperature=0.5
                )
                self.provider = provider
                self.model = model
                break
            except Exception as e:
                print(f"Failed to initialize {provider}: {e}")
                continue

        if self.llm is None:
            raise RuntimeError("Failed to initialize any LLM provider")

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer based on context and question"""
        prompt = f"""{self.system_prompt}

Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:"""

        return self.llm.generate(prompt)

    def chat(self, messages: list) -> str:
        """Chat with the LLM using a list of messages"""
        return self.llm.chat(messages)