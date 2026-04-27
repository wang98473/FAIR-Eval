from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hle_rebuild.common import FEATURE_COLUMNS, TASK_TYPE_MAP, count_empty_cells, require_columns

warnings.filterwarnings("ignore")

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    _NLP = None
    SPACY_OK = False


NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "without",
    "except", "hardly", "barely", "scarcely", "rarely", "seldom",
    "nothing", "nobody", "nowhere", "cannot", "can't", "won't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "shouldn't",
    "wouldn't", "couldn't", "mustn't",
}

CONDITIONAL_WORDS = {
    "if", "unless", "given", "assuming", "suppose", "supposed",
    "provided", "providing", "in case", "as long as", "only if",
    "whether", "when", "whenever", "should",
}

CONNECTIVE_WORDS = {
    "because", "since", "therefore", "thus", "hence", "so",
    "however", "although", "though", "despite", "nevertheless",
    "furthermore", "moreover", "consequently", "accordingly",
    "whereas", "while", "yet", "still", "otherwise", "instead",
    "meanwhile", "subsequently", "additionally", "finally",
    "first", "second", "third", "lastly", "in conclusion",
}

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "probably",
    "approximately", "roughly", "around", "about", "generally",
    "usually", "often", "sometimes", "occasionally", "tend",
    "tends", "likely", "unlikely", "suggest", "suggests",
    "indicate", "indicates", "appear", "appears", "seem", "seems",
    "somewhat", "rather", "fairly", "relatively", "typically",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "what", "which", "who",
    "whom", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "as", "up", "out", "if", "about",
}

TASK_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "calculation",
        re.compile(
            r"\b(how\s+many|how\s+much|calculat|comput|evaluat\s+the\s+expression"
            r"|solve|find\s+the\s+value|what\s+is\s+the\s+(sum|product|difference"
            r"|result|total|number|count|amount))\b"
            r"|\b\d+\s*[\+\-\*/]\s*\d+",
            re.I,
        ),
    ),
    (
        "metacognitive",
        re.compile(
            r"\b(best\s+way|most\s+appropriate|most\s+likely|most\s+effective"
            r"|most\s+suitable|best\s+describe|best\s+explain|evaluate|assess"
            r"|justify|critique|recommend)\b",
            re.I,
        ),
    ),
    (
        "reasoning",
        re.compile(
            r"^(why|how|what\s+would|what\s+might|what\s+could|what\s+causes"
            r"|explain|describe\s+how|how\s+does|how\s+do|how\s+would"
            r"|what\s+happens\s+if|what\s+will\s+happen)",
            re.I,
        ),
    ),
    ("judgment", re.compile(r"^(which|which\s+of|select|choose|identify\s+the)", re.I)),
    (
        "factual",
        re.compile(
            r"^(what|who|when|where|name|list|state|define|what\s+is|what\s+are"
            r"|who\s+is|who\s+was|when\s+did|where\s+is|where\s+did)",
            re.I,
        ),
    ),
]


def tokenize_words(text: str) -> list[str]:
    """Tokenize a string into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def count_sentences(text: str) -> int:
    """Count non-empty sentences using simple punctuation splitting."""
    parts = re.split(r"[.!?]+", text.strip())
    return sum(1 for part in parts if part.strip())


def content_words(text: str) -> set[str]:
    """Return non-trivial content words after stopword removal."""
    return {token for token in tokenize_words(text) if token not in STOPWORDS and len(token) > 1}


def feat_q_len(question: str) -> int:
    return len(question.split())


def feat_q_sent_count(question: str) -> int:
    return count_sentences(question)


def feat_q_negation_count(question: str) -> int:
    return sum(1 for token in tokenize_words(question) if token in NEGATION_WORDS)


def feat_q_conditional_count(question: str) -> int:
    return sum(1 for token in tokenize_words(question) if token in CONDITIONAL_WORDS)


def feat_q_avg_word_len(question: str) -> float:
    words = re.findall(r"[a-zA-Z]+", question)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def feat_q_subordinate_clause_count(doc) -> float:
    if doc is None:
        return float("nan")
    return sum(1 for token in doc if token.dep_ in {"advcl", "relcl", "ccomp"})


def feat_answer_len(answer_text: str) -> int:
    return len(answer_text.split())


def feat_answer_sent_count(answer_text: str) -> int:
    return count_sentences(answer_text)


def feat_answer_connective_density(answer_text: str) -> float:
    tokens = tokenize_words(answer_text)
    if not tokens:
        return 0.0
    count = sum(1 for token in tokens if token in CONNECTIVE_WORDS)
    return count / len(tokens)


def feat_answer_clause_count(doc) -> float:
    if doc is None:
        return float("nan")
    return sum(1 for token in doc if token.dep_ in {"advcl", "relcl", "ccomp"})


def feat_answer_entity_count(doc) -> float:
    if doc is None:
        return float("nan")
    return len(doc.ents)


def feat_answer_hedge_ratio(answer_text: str) -> float:
    tokens = tokenize_words(answer_text)
    if not tokens:
        return 0.0
    count = sum(1 for token in tokens if token in HEDGE_WORDS)
    return count / len(tokens)


def feat_answer_lexical_overlap(question: str, answer_text: str) -> float:
    question_words = content_words(question)
    answer_words = content_words(answer_text)
    if not question_words and not answer_words:
        return 0.0
    return len(question_words & answer_words) / len(question_words | answer_words)


def feat_answer_new_info_ratio(question: str, answer_text: str) -> float:
    question_words = content_words(question)
    answer_words = content_words(answer_text)
    if not answer_words:
        return 0.0
    return len(answer_words - question_words) / len(answer_words)


def feat_task_type(question: str) -> int:
    """Infer the task type from the question text."""
    stripped = question.strip()
    for task_name, pattern in TASK_RULES:
        if pattern.search(stripped):
            return TASK_TYPE_MAP[task_name]
    return TASK_TYPE_MAP["factual"]


def compute_row_features(question: str, rationale: str, q_doc=None, a_doc=None) -> dict[str, float]:
    """
    Compute the FAIR-Eval feature schema for one item.

    The legacy output schema keeps `answer_*` feature names, but in this HLE
    reconstruction workflow those values are derived from the `rationale` text.
    """
    return {
        "q_len": feat_q_len(question),
        "q_sent_count": feat_q_sent_count(question),
        "q_negation_count": feat_q_negation_count(question),
        "q_conditional_count": feat_q_conditional_count(question),
        "q_avg_word_len": feat_q_avg_word_len(question),
        "q_subordinate_clause_count": feat_q_subordinate_clause_count(q_doc),
        "answer_len": feat_answer_len(rationale),
        "answer_sent_count": feat_answer_sent_count(rationale),
        "answer_connective_density": feat_answer_connective_density(rationale),
        "answer_clause_count": feat_answer_clause_count(a_doc),
        "answer_entity_count": feat_answer_entity_count(a_doc),
        "answer_hedge_ratio": feat_answer_hedge_ratio(rationale),
        "answer_lexical_overlap": feat_answer_lexical_overlap(question, rationale),
        "answer_new_info_ratio": feat_answer_new_info_ratio(question, rationale),
        "task_type": feat_task_type(question),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FAIR-Eval-compatible HLE item features from local parquet input."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_with_embeddings.parquet"),
        help="Input parquet with question/rationale text and embedding columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_features.parquet"),
        help="Output parquet with derived feature columns",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    require_columns(
        df,
        ["id", "question", "rationale", "rationale_prompt", "question_embedding", "rationale_embedding"],
        "Feature input",
    )

    questions = df["question"].fillna("").astype(str).tolist()
    rationales = df["rationale"].fillna("").astype(str).tolist()

    if SPACY_OK:
        print(f"spaCy available: {SPACY_OK}")
        print(f"Parsing questions with spaCy: {len(questions)} rows")
        q_docs = list(_NLP.pipe(questions, batch_size=64))
        print(f"Parsing rationales with spaCy: {len(rationales)} rows")
        a_docs = list(_NLP.pipe(rationales, batch_size=64))
    else:
        print("spaCy model not available; spaCy-dependent features will be NaN")
        q_docs = [None] * len(questions)
        a_docs = [None] * len(rationales)

    records = [
        compute_row_features(question=question, rationale=rationale, q_doc=q_doc, a_doc=a_doc)
        for question, rationale, q_doc, a_doc in zip(questions, rationales, q_docs, a_docs)
    ]

    feature_df = pd.DataFrame(records)
    require_columns(feature_df, FEATURE_COLUMNS, "Generated feature frame")

    output_df = pd.concat([df, feature_df[FEATURE_COLUMNS]], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Rows written: {len(output_df)}")
    print(f"Columns written: {len(output_df.columns)}")
    print(f"task_type mapping: {TASK_TYPE_MAP}")

    empty_counts = count_empty_cells(output_df)
    print(f"Has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")


if __name__ == "__main__":
    main()
