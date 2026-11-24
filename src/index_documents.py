from src.rag import RAGIndex


def main() -> None:
    index = RAGIndex()
    count = index.build(force=True)
    print(
        f"{count} Textabschnitte aus "
        f"{sum(1 for _ in index.data_dir.rglob('*.pdf'))} PDFs indexiert."
    )
    print(f"LlamaIndex: {index.storage_dir}")


if __name__ == "__main__":
    main()
