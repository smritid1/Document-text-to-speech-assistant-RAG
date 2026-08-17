from src.settings import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.schema import ChunkMetadata
 
chunksize = settings.chunk_size
chunkoverlap = settings.chunk_overlap
 
 
def split_text_into_chunk(texts: list[tuple[int, str]], pdf_file: str):
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap
    )
 
    document = []
    metadatas = []
 
    for page_number, page_text in texts:
        page_chunk = splitter.split_text(page_text)
        for i, chunk in enumerate(page_chunk):
            chunk_id = f"{pdf_file}::p{page_number}::c{i}"
            document.append(chunk)
            metadata = ChunkMetadata(source=pdf_file, page=page_number, chunk_id=chunk_id)
            metadatas.append(metadata.model_dump())
 
    return document, metadatas






