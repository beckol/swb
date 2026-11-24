import os
from collections.abc import Iterable
from typing import Any, Literal

import ollama
from pydantic import BaseModel, PositiveInt


class TariffInfo(BaseModel):
    kunde_typ: Literal["Privatkunde", "Geschäftskunde"] | None = None
    stromverbrauch: PositiveInt | Literal["Keine Angabe"] = "Keine Angabe"
    ort: Literal["Bremen", "Bremerhaven", "Stuhr", "Weyhe", "Thedinghausen"] | None = None


SYSTEM_PROMPT = """
Du bist der digitale Stromtarifberater der swb.

Deine primäre Aufgabe ist die Erfassung folgender Kundeninformationen:

- Kundentyp (Privatkunde oder Geschäftskunde)
- Wohnort bzw. Versorgungsgebiet
- Jahresverbrauch in kWh

Regeln:
- Stelle immer nur eine Frage gleichzeitig.
- Frage ausschließlich nach noch fehlenden Informationen.
- Berücksichtige bereits genannte Kundeninformationen.
- Verändere keine bereits genannten Kundeninformationen eigenständig.
- Empfehle selbstständig keine Stromtarife. Die Tarifauswahl erfolgt durch eine separate Funktion.
- Beantworte fachliche Fragen ausschließlich anhand des bereitgestellten Dokumentenkontextes.
- Erfinde keine Tarife, Preise oder Vertragsbedingungen.
- Wenn der Dokumentenkontext keine ausreichenden Informationen enthält, teile dies dem Kunden mit.
- Sobald alle benötigten Informationen vorliegen, weise auf den Button „Tarif vorschlagen“ hin.
- Antworte kurz, freundlich und auf Deutsch.
""".strip()


class LLMChatService:
    """Ollama chat service used by the Streamlit application."""

    def __init__(
        self,
        model_id: str | None = None,
        ollama_host: str | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "OLLAMA_CHAT_MODEL",
            "llama3.2:3b",
        )
        self._client = ollama.Client(
            host=ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        context_text = ""
        if context:
            context_text = "\n\n".join(
                f"[Quelle: {item['source']}, Seite {item['page']}]\n{item['text']}"
                for item in context
            )

        conversation = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + (
                        "\n\nRelevante Dokumentenauszuege:\n"
                        + context_text
                        if context_text
                        else "\n\nEs wurden keine relevanten Dokumentenauszuege gefunden."
                    )
                ),
            },
            *[
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
                if message["role"] in {"user", "assistant"}
            ],
        ]

        response = self._client.chat(
            model=self.model_id,
            messages=conversation,
            options={
                "temperature": 0.2,
                "num_predict": 2000,
            },
        )
        return response.message.content

    def extract_tariff_info(
        self,
        messages: Iterable[dict[str, str]],
    ) -> dict[str, str | int | None]:
        conversation = [
            {
                "role": "system",
                "content": (
                    "Extrahiere ausschließlich Angaben, die der Kunde im "
                    "Gespräch ausdrücklich gemacht hat. Erfinde nichts. "
                    "kunde_typ ist Privatkunde oder Geschäftskunde, sonst null. "
                    "stromverbrauch ist ein positiver Wert in kWh oder "
                    "'Keine Angabe'. ort ist Bremen, Bremerhaven, Stuhr, Weyhe "
                    "oder Thedinghausen, sonst null. Gib nur das JSON gemäß Schema aus."
                ),
            },
            *[
                {"role": message["role"], "content": message["content"]}
                for message in messages
                if message["role"] in {"user", "assistant"}
            ],
        ]
        response = self._client.chat(
            model=self.model_id,
            messages=conversation,
            format=TariffInfo.model_json_schema(),
            options={"temperature": 0},
        )
        return TariffInfo.model_validate_json(response.message.content).model_dump()
