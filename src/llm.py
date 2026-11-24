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
Du bist der freundliche und kompetente Stromtarifberater der swb.
Führe das Gespräch natürlich, freundlich, hilfsbereit und auf Deutsch.
Nutze ausschließlich die bereitgestellten Dokumentenauszüge für konkrete
Tarif- und Produktinformationen. Wenn die Angaben für eine konkrete 
Beratung noch fehlen, stelle genau eine kurze, relevante Folgefrage.

Wir wollen den Kunden ein Tarif empfehlen, der zu seinen Bedürfnissen passt.
Dafür musst du herausfinden, ob der Kunde einen Tarif für Privatkunden oder einen Geschäftskunden sucht.
Wie hoch sein Stromverbrauch etwa ist und ob er in Bremen, Bremerhaven oder (Stuhr, Weyhe oder Thedinghausen)
wohnt.

Stelle gezielt fragen, um die relevanten Informationen zu erhalten. Vergiss keine Informationen die du bereits vom Kunden erhalten hast.

In (Stuhr, Weyhe oder Thedinghausen) gibt es nur diese Verträge:
['swb Strom von hier', 'swb Strom spar S', 'swb Strom spar L', 
'swb Strom Business', 'swb Strom dynamisch', 'swb Strom RLM']

In Bremen und Bremerhaven gibt es nur diese Verträge:
['swb Strom basis', 'swb Strom dynamisch', 'swb Strom Ersatzversorgung',
'swb Strom von hier', 'swb Strom spar S', 'swb Strom spar L',
'swb Strom Business', 'swb Strom dynamisch', 'swb Strom RLM']

Gib im normalen Gespräch kein JSON aus. Das JSON für einen Tarifvorschlag
wird separat über die entsprechende Funktion in der Anwendung erstellt.

Hat der Kunde eine allgemeine Frage zu den Stromtarifen, dann beantworte sie ihn. Du beratest nur zu Stromtarifen.
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
