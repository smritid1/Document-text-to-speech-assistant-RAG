import os

from src.pdf_loader import read_pdf
from src.chunking import split_text_into_chunk
from src.vector_database import create_vectordb, load_vectordb
from src.rag import ask_question
from src.bm25_index import build_bm25_index
from src.hybrid_retriever import hybrid_retrieve
from Day_4.src.settings import settings
from src.schema import ChatMessage, ChatRole


# db_exist = os.path.exists(settings.persist_directory) and os.listdir(settings.persist_directory)


def main():
    pdf_path = "food.pdf"

    # db_exists = os.path.exists(settings.persist_directory) and os.listdir(settings.persist_directory)

    # if db_exists:
    #     print("Found an existing vector database. Loading it...")
    #     vectordb = load_vectordb()
    #     # NOTE: if you re-ingest a different PDF later while an old database
    #     # still exists, this branch will skip rebuilding the BM25 index too --
    #     # see the callout below.
    # else:
    pages = read_pdf(pdf_path)
    texts, metadatas = split_text_into_chunk(pages, pdf_path)

    for i, meta in enumerate(metadatas):
        meta["chunk_id"] = f"{meta.get('source', pdf_path)}_p{meta.get('page')}_{i}"

    vectordb = create_vectordb(texts, metadatas)   # <-- this was missing
    build_bm25_index(texts, metadatas)

    print("\nReady! Ask questions about your PDF. Type 'exit' to quit.\n")

    history: list[ChatMessage] = []
    
    while True:
        question = input("User: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = ask_question(vectordb, question,history)

        print(f"\nBot: {answer.answer}")
        if answer.pages:
            print(f"(Source pages: {answer.pages})")
        print()


        history.append(ChatMessage(role=ChatRole.USER, content=answer.question))
        history.append(ChatMessage(role=ChatRole.ASSISTANT, content=answer.answer))


if __name__ == "__main__":
    main()