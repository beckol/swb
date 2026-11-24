import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm import LLMChatService


class LLMChatServiceTests(unittest.TestCase):
    def test_chat_extracts_customer_info_before_sending_prompt(self) -> None:
        events: list[str] = []
        history = [
            {"role": "user", "content": "Ich wohne in Bremen."},
            {"role": "assistant", "content": "Wie hoch ist Ihr Verbrauch?"},
            {"role": "user", "content": "3.200 kWh."},
        ]
        tariff_info = {
            "kunde_typ": "Privatkunde",
            "stromverbrauch": 3200,
            "ort": "Bremen",
        }
        response = SimpleNamespace(
            message=SimpleNamespace(content="Danke für die Angaben.")
        )
        service = LLMChatService()

        def extract_info(
            input_messages: list[dict[str, str]],
        ) -> dict[str, str | int | None]:
            events.append("extract")
            self.assertEqual(input_messages, history)
            return tariff_info

        def send_chat(
            *,
            model: str,
            messages: list[dict[str, str]],
            options: dict[str, int | float],
        ) -> SimpleNamespace:
            events.append("chat")
            self.assertIn(
                json.dumps(tariff_info, ensure_ascii=False),
                messages[0]["content"],
            )
            self.assertEqual(messages[-1], history[-1])
            return response

        with (
            patch.object(service, "extract_tariff_info", side_effect=extract_info),
            patch.object(service._client, "chat", side_effect=send_chat),
        ):
            result = service.chat(history)

        self.assertEqual(events, ["extract", "chat"])
        self.assertEqual(result, "Danke für die Angaben.")


if __name__ == "__main__":
    unittest.main()
