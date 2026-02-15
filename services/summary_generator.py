import re

def simple_summary(text: str, max_sentences=3):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if len(s.split()) > 6]

    if not sentences:
        return "Summary not available."

    return " ".join(sentences[:max_sentences])
