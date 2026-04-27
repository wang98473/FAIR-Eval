from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from hle_rebuild.common import HLE_DATASET_DESCRIPTION, count_empty_cells, require_columns


def build_client(api_key: str, base_url: str | None) -> OpenAI:
    """Build an OpenAI-compatible client from environment configuration."""
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def resolve_client_settings(
    model_override: str | None,
    dimensions_override: int | None,
) -> tuple[OpenAI, str, int | None]:
    """Resolve API configuration from environment variables and CLI overrides."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required to generate embeddings")

    base_url = os.environ.get("OPENAI_BASE_URL")
    model = model_override or os.environ.get("OPENAI_EMBEDDING_MODEL")
    if not model:
        raise EnvironmentError(
            "Provide an embedding model via --model or OPENAI_EMBEDDING_MODEL"
        )

    dimensions = dimensions_override
    if dimensions is None and os.environ.get("OPENAI_EMBEDDING_DIMENSIONS"):
        dimensions = int(os.environ["OPENAI_EMBEDDING_DIMENSIONS"])

    client = build_client(api_key=api_key, base_url=base_url)
    return client, model, dimensions


def create_embeddings_batch(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimensions: int | None,
) -> list[str | None]:
    """Create one batch of embeddings and serialize them as JSON strings."""
    kwargs: dict[str, object] = {"input": texts, "model": model}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    try:
        response = client.embeddings.create(**kwargs)
        return [json.dumps(item.embedding, ensure_ascii=False) for item in response.data]
    except Exception as batch_exc:
        print(f"Batch request failed; falling back to single-item retries: {batch_exc}")
        outputs: list[str | None] = []
        for text in texts:
            try:
                single_response = client.embeddings.create(**{**kwargs, "input": [text]})
                outputs.append(
                    json.dumps(single_response.data[0].embedding, ensure_ascii=False)
                )
            except Exception as single_exc:
                preview = text[:80].replace("\n", " ")
                print(f"  Failed to embed text: {preview!r} | error: {single_exc}")
                outputs.append(None)
        return outputs


def build_embeddings_with_progress(
    client: OpenAI,
    texts: list[str],
    model: str,
    dimensions: int | None,
    batch_size: int,
    label: str,
) -> list[str | None]:
    """Create embeddings in batches while printing progress."""
    outputs: list[str | None] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        outputs.extend(create_embeddings_batch(client, batch, model, dimensions))
        current = min(start + batch_size, total)
        print(f"{label}: {current}/{total}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local HLE subset embeddings using an OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_raw.parquet"),
        help="Input parquet file produced by extract_hle_subset.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hle_rebuild/items/hle_subset_with_embeddings.parquet"),
        help="Output parquet file with question and rationale embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name. Overrides OPENAI_EMBEDDING_MODEL when provided.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional embedding dimension override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding request batch size",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    require_columns(df, ["question", "rationale", "rationale_prompt"], "Embedding input")

    client, model, dimensions = resolve_client_settings(args.model, args.dimensions)

    question_texts = [
        f"{HLE_DATASET_DESCRIPTION} ### QUESTION: {text}"
        for text in df["rationale_prompt"].fillna("").astype(str).tolist()
    ]
    rationale_texts = df["rationale"].fillna("").astype(str).tolist()

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {model}")
    print(f"Dimensions override: {dimensions}")

    df["question_embedding"] = build_embeddings_with_progress(
        client=client,
        texts=question_texts,
        model=model,
        dimensions=dimensions,
        batch_size=args.batch_size,
        label="question_embedding",
    )
    df["rationale_embedding"] = build_embeddings_with_progress(
        client=client,
        texts=rationale_texts,
        model=model,
        dimensions=dimensions,
        batch_size=args.batch_size,
        label="rationale_embedding",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"Rows written: {len(df)}")
    print(f"question_embedding non-null: {int(df['question_embedding'].notna().sum())}")
    print(f"rationale_embedding non-null: {int(df['rationale_embedding'].notna().sum())}")

    empty_counts = count_empty_cells(df)
    print(f"Has empty cells: {bool(empty_counts)}")
    if empty_counts:
        print("Empty-cell counts by column:")
        for column, count in empty_counts.items():
            print(f"  {column}: {count}")


if __name__ == "__main__":
    main()
