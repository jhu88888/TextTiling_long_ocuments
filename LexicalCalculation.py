from nltk.tokenize.texttiling import TextTilingTokenizer
from typing import List

def clean_tiling_output(tt_text: str) -> str:
    cleaned_text = tt_text.replace("\n\n", "")
    return cleaned_text

def prepare_tiling_input(text: str) -> str:
    return text.replace("", "\n\n")  # Preserving this as in your original, even though it does nothing.

def predict_chunk_boundaries(
    document: str,
    encoder: tiktoken.core.Encoding,
    w: int = 40,
    k: int = 20
) -> List[int]:
    tokenizer = TextTilingTokenizer(w=w, k=k)
    tiled_sections = tokenizer.tokenize(prepare_tiling_input(document))

    start_index = 0
    chunk_start_positions = [start_index]

    for section in tiled_sections:
        processed_section = clean_tiling_output(section)
        start_index += len(encoder.encode(processed_section, allowed_special="all"))
        chunk_start_positions.append(start_index)

    return chunk_start_positions
