from typing import List

from tqdm import tqdm

from src.indexer import (Chunk, ChunkIndex, chunk_markdown, chunk_python,
                         extract_files)
from src.io_utils import DataLoadError, read_text, save_model

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MAX_CHUNK_SIZE = 2000


class RagCLI():
    def __init__(self) -> None:
        pass

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
        """Chunk every file of data/raw and save the index in data/processed.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
        """
        if not 0 < max_chunk_size <= MAX_CHUNK_SIZE:
            print(f"Error: --max_chunk_size must be between 1 and "
                  f"{MAX_CHUNK_SIZE}, got {max_chunk_size}")
            return
        all_chunks: List[Chunk] = []
        for path in tqdm(extract_files(RAW_DIR), desc="Indexing"):
            try:
                text = read_text(path)
            except DataLoadError as e:
                tqdm.write(f"Skipped: {e}")
                continue
            chunker = chunk_python if path.endswith(".py") else chunk_markdown
            all_chunks += chunker(path, text, max_size=max_chunk_size)
        try:
            save_model(ChunkIndex(max_chunk_size=max_chunk_size,
                                  chunks=all_chunks),
                       PROCESSED_DIR, "chunks.json")
        except DataLoadError as e:
            print(f"Error: {e}")
            return
        print(f"Ingestion complete! {len(all_chunks)} chunks saved under "
              f"{PROCESSED_DIR}/")

    def search(self):
        pass

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
