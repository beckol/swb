import os
from pathlib import Path
from typing import Any

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.ollama import OllamaEmbedding
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


class RAGIndex:
    def __init__(
        self,
        data_dir: Path | str = ROOT / "data",
        storage_dir: Path | str = ROOT / "data" / ".llamaindex",
        ollama_host: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model
            or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=ollama_host or os.getenv(
                "OLLAMA_HOST", "http://localhost:11434"
            ),
        )
        self.index: VectorStoreIndex | None = None

    def build(self, force: bool = False) -> int:
        if self.index is not None and not force:
            return len(self.index.index_struct.nodes_dict)
        if not force and (self.storage_dir / "index_store.json").exists():
            self.index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(self.storage_dir)),
                embed_model=self.embed_model,
            )
            return len(self.index.index_struct.nodes_dict)

        documents = [
            Document(
                text=text,
                metadata={"source": str(pdf.relative_to(self.data_dir)), "page": n},
            )
            for pdf in self.data_dir.rglob("*.pdf")
            for n, page in enumerate(PdfReader(pdf).pages, 1)
            for text in [page.extract_text() or ""]
            if text.strip()
        ]
        if not documents:
            raise ValueError(f"Keine PDF-Texte gefunden in {self.data_dir}")

        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model,
            transformations=[TokenTextSplitter(chunk_size=1024, chunk_overlap=200)],
        )
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        return len(self.index.index_struct.nodes_dict)

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        self.build()
        if self.index is None:
            return []
        return [
            {
                "text": item.node.get_content(metadata_mode=MetadataMode.NONE),
                **item.node.metadata,
            }
            for item in self.index.as_retriever(similarity_top_k=limit).retrieve(query)
        ]
