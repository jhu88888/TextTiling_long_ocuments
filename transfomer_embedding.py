from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize.texttiling import TextTilingTokenizer

def predict_chunk_boundaries(
    document: str,
    encoder: tiktoken.core.Encoding,
    sentence_delimiter: int,
    model: SentenceTransformer,
) -> List[int]:
    # Encode the delimiter and assert a single token representation
    encoded_delimiter = encoder.encode(
        sentence_delimiter, allowed_special={sentence_delimiter}
    )
    assert len(encoded_delimiter) == 1

    # Encode the entire document
    encoded_document = encoder.encode(
        document, allowed_special={sentence_delimiter}
    )

    # Identify potential starting points right after each delimiter
    start_points = [
        idx + 1 for idx, token in enumerate(encoded_document)
        if token == encoded_delimiter[0]
    ]

    # Split the document at each delimiter to form sentences
    sentences = [
        part for part in document.split(sentence_delimiter) if part
    ]

    # Combine adjacent sentences to form pairs for comparison
    combined_sentences = [
        sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)
    ]
    if len(sentences) % 2 != 0:
        combined_sentences.append(sentences[-1])

    # Filter start_points for even indices
    filtered_start_points = [
        point for idx, point in enumerate(start_points) if idx % 2 == 0
    ]

    # Calculate cosine similarity between consecutive sentence pairs
    sentence_embeddings = model.encode(combined_sentences)
    similarity_scores = [
        np.dot(sentence_embeddings[i], sentence_embeddings[i + 1])
        for i in range(len(sentence_embeddings) - 1)
    ]

    # Smoothing the similarity scores
    window_length = min(len(combined_sentences) // 9, 11)
    smoothed_scores = np.convolve(similarity_scores, np.ones(window_length) / window_length, 'same')

    # Identify boundaries using the TextTiling algorithm
    tokenizer = TextTilingTokenizer()
    depth_scores = tokenizer._depth_scores(smoothed_scores)
    boundaries = tokenizer._identify_boundaries(depth_scores)

    # Determine the starting points of new segments
    return [0] + [
        point for point, boundary in zip(filtered_start_points, boundaries)
        if boundary > 0
    ]
