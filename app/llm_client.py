import anthropic

from app.config import settings

SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы строго на основе предоставленных фрагментов документов.

Правила:
- Отвечай только на основе текста в присланных фрагментах. Если ответа там нет — честно скажи, что в базе знаний нет информации по этому вопросу.
- Не выдумывай факты, которых нет в тексте.
- Отвечай на том же языке, на котором задан вопрос.
- Указывай, из какого источника (source) взята информация, если это уместно.
"""


class LLMClientError(Exception):
    pass


class LLMUnavailableError(LLMClientError):
    pass


def _extract_text(message: anthropic.types.Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text").strip()


def answer_question(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "В базе знаний пока нет ни одного документа — сначала загрузите файл через /upload."

    if not settings.anthropic_api_key:
        # Честный fallback: без ключа LLM недоступен, но поиск релевантных
        # фрагментов (эмбеддинги) работает полностью локально и бесплатно.
        # Вместо ошибки показываем сам найденный контекст — это доказывает,
        # что поисковая часть RAG работает корректно.
        top = chunks[0]
        return (
            "[Демо-режим без API-ключа] LLM недоступен, но поиск по базе сработал.\n"
            f"Наиболее релевантный найденный фрагмент (источник: {top['source']}):\n\n"
            f"{top['text']}"
        )

    context = "\n\n".join(
        f"[Источник: {c['source']}, фрагмент {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        message = client.messages.create(
            model=settings.model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Фрагменты документов:\n\n{context}\n\nВопрос: {question}",
                }
            ],
        )
    except anthropic.APIConnectionError as e:
        raise LLMUnavailableError(f"Не удалось связаться с LLM API: {e}") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailableError(f"Превышен лимит запросов к LLM API: {e}") from e
    except anthropic.APIStatusError as e:
        raise LLMUnavailableError(f"LLM API вернул ошибку {e.status_code}: {e.message}") from e

    return _extract_text(message)
