
import re
import html

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()

# Preserve negation words
KEEP_WORDS = {"no", "not", "nor", "never"}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS


def normalize_text(text):

    text = html.unescape(str(text))

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Normalize URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " urltoken ",
        text
    )

    # 4. Normalize email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " emailtoken ",
        text
    )

    # 5. Normalize numbers
    text = re.sub(
        r"\b\d+\b",
        " numbertoken ",
        text
    )

    # 6. Remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # 7. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(text):

    normalized = normalize_text(text)

    # 8. Tokenization
    tokens = normalized.split()

    cleaned_tokens = []

    for word in tokens:

        # 9. Stop-word removal
        if word in STOP_WORDS:
            continue

        # 10. Stemming
        word = stemmer.stem(word)

        cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)


def clean_batch(texts, batch_size=500):

    return [
        clean_text(text)
        for text in texts
    ]


if __name__ == "__main__":

    sample = (
        "CONGRATULATIONS!!! You have WON a FREE prize. "
        "Visit https://example.com. This is NOT a test."
    )

    print("Original:", sample)

    print("Normalized:", normalize_text(sample))

    print("Cleaned:", clean_text(sample))
