from typing import List


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Simple text splitter:
    - Takes a long string
    - Returns a list of overlapping chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap  # move with overlap

    return chunks


if __name__ == "__main__":
    sample_text = (
        "Machine learning is a subset of artificial intelligence that focuses "
        "on building systems that learn from data. " * 20
    )
    chunks = split_text_into_chunks(sample_text, chunk_size=100, chunk_overlap=20)
    print(f"Total chunks: {len(chunks)}")
    for i, ch in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---\n{ch}")
