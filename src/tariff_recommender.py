import json
from pathlib import Path
from typing import Any, Literal

CustomerType = Literal["Privatkunde", "Geschäftskunde"]
Consumption = int | Literal["Keine Angabe"] | None
Region = Literal["Bremen", "Bremerhaven", "Stuhr", "Weyhe", "Thedinghausen"]

_CUSTOMER_TYPE_MAP = {
    "Privatkunde": "Privatperson",
    "Geschäftskunde": "Business",
}
_TARIFFS_PATH = Path(__file__).resolve().parents[1] / "data" / "stromtarif.json"


def recommend_tariffs(
    kunde_typ: CustomerType | None,
    stromverbrauch: Consumption,
    ort: Region | None,
) -> list[dict[str, Any]]:
    if kunde_typ is None or ort is None:
        return []
    if kunde_typ not in _CUSTOMER_TYPE_MAP:
        raise ValueError(f"Unbekannter Kundentyp: {kunde_typ}")
    if stromverbrauch is not None and stromverbrauch != "Keine Angabe":
        if (
            not isinstance(stromverbrauch, int)
            or isinstance(stromverbrauch, bool)
            or stromverbrauch <= 0
        ):
            raise ValueError("Der Stromverbrauch muss eine positive kWh-Zahl sein.")

    customer_type = _CUSTOMER_TYPE_MAP[kunde_typ]
    tariffs = json.loads(_TARIFFS_PATH.read_text(encoding="utf-8"))["tariffs"]
    matches = [
        tariff
        for tariff in tariffs
        if tariff["region"] == ort
        and customer_type in tariff["customer_types"]
        and _matches_consumption(tariff["consumption_range"], stromverbrauch)
    ]
    return matches[:3]


def _matches_consumption(value: str, consumption: Consumption) -> bool:
    if value == "Keine Grenze" or consumption is None or consumption == "Keine Angabe":
        return True

    numbers = [
        int(part.replace(".", ""))
        for part in value.split()
        if part.replace(".", "").isdigit()
    ]
    if not numbers:
        raise ValueError(f"Unbekanntes Verbrauchsintervall: {value}")
    if value.startswith("bis "):
        return consumption <= numbers[-1]
    if len(numbers) != 2:
        raise ValueError(f"Unbekanntes Verbrauchsintervall: {value}")
    return numbers[0] <= consumption <= numbers[1]
