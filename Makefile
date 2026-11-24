.PHONY: help install run dev check index index-reset ollama-check ollama-pull

help:
	@echo "Verfuegbare Befehle:"
	@echo "  make run      Startet die Streamlit-Anwendung"
	@echo "  make install  Installiert die Projektabhaengigkeiten"
	@echo "  make dev      Startet die Anwendung im Entwicklungsmodus"
	@echo "  make check    Prueft die Python-Dateien auf Syntaxfehler"
	@echo "  make index    Indexiert alle PDFs aus data/"
	@echo "  make index-reset Loescht und erstellt den LlamaIndex-Index neu"
	@echo "  make ollama-check  Prueft Ollama und zeigt installierte Modelle"
	@echo "  make ollama-pull   Laedt das Standard-Chatmodell und das Embedding-Modell"

install:
	uv sync

dev:
	uv run streamlit run streamlit/app.py --server.runOnSave true

check:
	uv run python -m compileall -q src streamlit

run:
	uv run streamlit run streamlit/app.py

index:
	uv run python -m src.index_documents

index-reset:
	rm -rf data/.llamaindex
	uv run python -m src.index_documents

ollama-check:
	@curl --fail --silent http://localhost:11434/api/tags >/dev/null || \
		(echo "Ollama laeuft nicht unter http://localhost:11434"; exit 1)
	@ollama list

ollama-pull:
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text
