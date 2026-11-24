import html
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm import LLMChatService, TariffInfo
from src.rag import RAGIndex
from src.tariff_recommender import recommend_tariffs


def _render_tariff_cards(tariffs: list[dict[str, object]]) -> str:
    cards = []
    recommendation_index = (len(tariffs) - 1) // 2
    for index, tariff in enumerate(tariffs):
        prices = str(tariff["prices"])
        base_price, separator, consumption_price = prices.partition(
            "; Verbrauchspreis:"
        )
        base_price = base_price.removeprefix("Grundpreis:").strip()
        consumption_price = consumption_price.strip() if separator else ""

        def lines(value: str) -> str:
            return "<br>".join(
                html.escape(line.strip())
                for line in value.split(";")
                if line.strip()
            )

        consumption = str(tariff["consumption_range"])
        if consumption == "Keine Grenze":
            consumption = "Keine Begrenzung"
        else:
            consumption += " kWh/Jahr"

        recommended = index == recommendation_index
        card_class = " tariff-card--recommended" if recommended else ""
        badge = (
            '<div class="tariff-card__badge">Unsere Empfehlung</div>'
            if recommended
            else ""
        )
        cards.append(
            f'<article class="tariff-card{card_class}">{badge}'
            '<div class="tariff-card__content">'
            '<div class="tariff-card__leaf" aria-hidden="true">🌿</div>'
            f'<h3 class="tariff-card__title">{html.escape(str(tariff["name"]))}</h3>'
            '<div class="tariff-card__consumption"><span>Verbrauch</span>'
            f'<strong>{html.escape(consumption)}</strong></div>'
            '<div class="tariff-card__price-section">'
            '<span class="tariff-card__label">Grundpreis</span>'
            f'<div class="tariff-card__price">{lines(base_price)}</div></div>'
            '<div class="tariff-card__price-section">'
            '<span class="tariff-card__label">Verbrauchspreis</span>'
            f'<div class="tariff-card__price">{lines(consumption_price)}</div></div>'
            '<p class="tariff-card__description">'
            f'{html.escape(str(tariff["description"]))}</p>'
            '</div></article>'
        )
    return '<div class="tariff-grid">' + "".join(cards) + "</div>"


st.set_page_config(
    page_title="swb Stromtarifberater",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f7f7;
        color: #333333;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6e6e6;
    }
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        margin-bottom: 8px;
        padding: 8px 12px;
    }
    [data-testid="stChatInput"] {
        border-radius: 8px;
    }
    .stApp button[data-testid="stBaseButton-primary"] {
        background: #e30613;
        border-color: #e30613;
        border-radius: 999px;
        color: #ffffff;
        font-weight: 700;
    }
    .stApp button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        background: #c90010;
        border-color: #c90010;
    }
    .stApp button[data-testid="stBaseButton-primary"]:disabled {
        background: #e30613;
        border-color: #e30613;
        opacity: 0.55;
    }
    .swb-masthead {
        display: flex;
        align-items: center;
        gap: 20px;
        min-height: 64px;
        margin-bottom: 2rem;
        padding: 12px 20px;
        background: #ffffff;
        border-top: 4px solid #e30613;
        border-bottom: 1px solid #e6e6e6;
    }
    .swb-logo {
        color: #e30613;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -2.5px;
        line-height: 1;
    }
    .swb-masthead__divider {
        width: 1px;
        height: 34px;
        background: #dedede;
    }
    .swb-masthead__section {
        color: #333333;
        font-size: 1rem;
        font-weight: 700;
    }
    .swb-masthead__label {
        margin-top: 3px;
        color: #666666;
        font-size: 0.8rem;
    }
    .swb-masthead__tag {
        margin-left: auto;
        padding: 7px 12px;
        border-radius: 999px;
        background: #fff1f2;
        color: #c90010;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .sidebar-heading {
        color: #e30613;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .tariff-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        align-items: stretch;
        gap: 24px;
        max-width: 1240px;
        margin: 24px auto;
    }
    .tariff-card {
        position: relative;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
    }
    .tariff-card--recommended {
        border-color: #e30613;
        box-shadow: 0 8px 24px rgba(227, 6, 19, 0.14);
        transform: translateY(-8px);
    }
    .tariff-card__badge {
        padding: 10px 12px;
        background: #e30613;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .tariff-card__content {
        padding: 24px 20px;
    }
    .tariff-card__leaf {
        color: #61b99d;
        font-size: 26px;
        line-height: 1;
    }
    .tariff-card__title {
        margin: 12px 0 18px;
        color: #e30613;
        font-size: 1.5rem;
        line-height: 1.25;
    }
    .tariff-card__consumption {
        display: grid;
        gap: 4px;
        margin-bottom: 22px;
        color: #242424;
    }
    .tariff-card__consumption span,
    .tariff-card__label {
        color: #5c5c5c;
        font-size: 0.85rem;
    }
    .tariff-card__price-section {
        margin-top: 16px;
    }
    .tariff-card__price {
        margin-top: 4px;
        color: #333333;
        font-size: 0.95rem;
        font-weight: 650;
        line-height: 1.55;
    }
    .tariff-card__description {
        margin: 20px 0 0;
        color: #626262;
        font-size: 0.86rem;
        line-height: 1.5;
    }
    @media (max-width: 700px) {
        .swb-masthead {
            gap: 12px;
            padding: 12px;
        }
        .swb-logo {
            font-size: 28px;
        }
        .swb-masthead__tag {
            font-size: 0.72rem;
            padding: 6px 9px;
        }
        .tariff-grid {
            grid-template-columns: 1fr;
            max-width: 440px;
        }
        .tariff-card--recommended {
            transform: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="swb-masthead">
        <div class="swb-logo" role="img" aria-label="swb">swb</div>
        <div class="swb-masthead__divider"></div>
        <div>
            <div class="swb-masthead__section">Strom</div>
            <div class="swb-masthead__label">Tarifberatung</div>
        </div>
        <div class="swb-masthead__tag">KI-Tarifberater</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-heading">Beratungsassistent</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("KI-Assistent")
    st.write("**Ollama (lokal)**")
    st.caption("RAG-Quellen: `data/`")
    st.caption("Neue Sitzung startet einen neuen Chat.")
    if st.button("Chat zurücksetzen", use_container_width=True):
        st.session_state.pop("tariff_info", None)
        st.session_state.pop("recommended_tariffs", None)
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hallo! Ich helfe Ihnen gerne bei der Auswahl eines "
                    "passenden Stromtarifs. Wie kann ich Ihnen helfen?"
                ),
            }
        ]
        st.rerun()

st.title("swb Stromtarifberater")
st.write("Beschreiben Sie Ihren Bedarf – ich stelle Ihnen die nötigen Fragen.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hallo! Ich helfe Ihnen gerne bei der Auswahl eines "
                "passenden Stromtarifs. Wie kann ich Ihnen helfen?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            source_labels = sorted(
                {
                    f"{source['source']} (Seite {source['page']})"
                    for source in message["sources"]
                }
            )
            st.caption("Quellen: " + ", ".join(source_labels))

if prompt := st.chat_input("Ihre Nachricht"):
    st.session_state.pop("tariff_info", None)
    st.session_state.pop("recommended_tariffs", None)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if "chat_service" not in st.session_state:
            st.session_state.chat_service = LLMChatService()
        if "rag_index" not in st.session_state:
            st.session_state.rag_index = RAGIndex()

        context = st.session_state.rag_index.retrieve(prompt, limit=5)
        response = st.session_state.chat_service.chat(
            st.session_state.messages,
            context=context,
        )
        sources = [
            {
                "source": item["source"],
                "page": item["page"],
            }
            for item in context
        ]
    except Exception as error:
        response = (
            "Die Verbindung zum KI-Service konnte nicht hergestellt werden. "
            f"Technische Meldung: {error}"
        )
        sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(response)
        if sources:
            source_labels = sorted(
                {
                    f"{source['source']} (Seite {source['page']})"
                    for source in sources
                }
            )
            st.caption("Quellen: " + ", ".join(source_labels))

if st.button(
    "Tarif vorschlagen",
    type="primary",
    disabled=not any(
        message["role"] == "user" for message in st.session_state.messages
    ),
):
    st.session_state.pop("tariff_info", None)
    st.session_state.pop("recommended_tariffs", None)
    try:
        if "chat_service" not in st.session_state:
            st.session_state.chat_service = LLMChatService()
        extracted_info = (
            st.session_state.chat_service.extract_tariff_info(
                st.session_state.messages
            )
        )
        tariff_info = TariffInfo.model_validate(extracted_info)
        st.session_state.recommended_tariffs = recommend_tariffs(
            tariff_info.kunde_typ,
            tariff_info.stromverbrauch,
            tariff_info.ort,
        )
    except Exception as error:
        st.error(f"Die Tarifvorschläge konnten nicht erstellt werden: {error}")

if "recommended_tariffs" in st.session_state:
    st.subheader("Tarifvorschläge")
    if not st.session_state.recommended_tariffs:
        st.info("Mit den Gesprächsinformationen konnte kein passender Tarif gefunden werden.")
    else:
        st.markdown(
            _render_tariff_cards(st.session_state.recommended_tariffs),
            unsafe_allow_html=True,
        )
