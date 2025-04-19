import nltk
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config['data']['chunk_size']
        self.overlap = config['data']['overlap']
        nltk.download('punkt')

    def process_text(self, text: str) -> List[str]:
        return self.nltk_based_splitter(text, self.chunk_size, self.overlap)

    @staticmethod
    def nltk_based_splitter(text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        for value in text.values():
            sentences = nltk.sent_tokenize(value)
            
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            if overlap > 0:
                overlapping_chunks = []
                for i in range(len(chunks)):
                    if i > 0:
                        start_overlap = max(0, len(chunks[i-1]) - overlap)
                        chunk_with_overlap = chunks[i-1][start_overlap:] + " " + chunks[i]
                        overlapping_chunks.append(chunk_with_overlap[:chunk_size])
                    else:
                        overlapping_chunks.append(chunks[i][:chunk_size])
                return overlapping_chunks

        return chunks