import os
from pathlib import Path

# Set environment variable before importing nltk
LOCAL_NLTK_DIR = Path(__file__).resolve().parent.parent / ".nltk_data"
LOCAL_NLTK_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
LOCAL_NLTK_DIR.chmod(0o755)
os.environ["NLTK_DATA"] = str(LOCAL_NLTK_DIR)

import argparse
import math
import re
from collections import Counter

import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def get_stemmed_stopwords(stemmer: PorterStemmer, languages: list[str]) -> set[str]:
    nltk.download("stopwords", download_dir=str(LOCAL_NLTK_DIR), quiet=True)
    raw_stops: set[str] = set()
    for lang in languages:
        raw_stops.update(stopwords.words(lang))
    stemmed_stops = {stemmer.stem(w) for w in raw_stops}
    return raw_stops | stemmed_stops


def load_nltk_background_counts(
    corpus_name: str, stemmer: PorterStemmer, stop_stems: set[str]
) -> tuple[Counter[str], int]:
    nltk.download(corpus_name, download_dir=str(LOCAL_NLTK_DIR), quiet=True)
    corpus_module = getattr(nltk.corpus, corpus_name)

    neg_counts: Counter[str] = Counter()
    total_neg_tokens = 0

    for w in corpus_module.words():
        w_low = w.lower()
        if len(w_low) < 2 or not w_low.isalpha() or w_low in stop_stems:
            continue
        stem = stemmer.stem(w_low)
        if stem not in stop_stems:
            neg_counts[stem] += 1
            total_neg_tokens += 1

    return neg_counts, total_neg_tokens


def extract_and_stem_tokens(
    file_path: Path, stemmer: PorterStemmer, stop_stems: set[str]
) -> list[str]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())

    tokens: list[str] = []
    for w in words:
        if w in stop_stems:
            continue
        stem = stemmer.stem(w)
        if stem not in stop_stems:
            tokens.append(stem)
    return tokens


def load_corpus_documents(
    corpus_dir: Path, stemmer: PorterStemmer, stop_stems: set[str]
) -> list[list[str]]:
    docs: list[list[str]] = []
    for file_path in corpus_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt"}:
            tokens = extract_and_stem_tokens(file_path, stemmer, stop_stems)
            if tokens:
                docs.append(tokens)
    return docs


def build_valid_vocabulary(
    target_docs: list[list[str]],
    neg_counts: Counter[str],
    total_neg_tokens: int,
    max_df_ratio: float,
    min_count: int,
    contrast_ratio: float,
) -> set[str]:
    num_docs = len(target_docs)
    if num_docs == 0:
        return set()

    target_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    total_target_tokens = 0

    for doc in target_docs:
        target_counts.update(doc)
        doc_freq.update(set(doc))
        total_target_tokens += len(doc)

    valid_vocab: set[str] = set()
    smoothing = 1e-6

    for stem, count in target_counts.items():
        if count < min_count:
            continue

        if num_docs > 1 and (doc_freq[stem] / num_docs) > max_df_ratio:
            continue

        if total_neg_tokens > 0:
            p_target = count / total_target_tokens
            p_neg = neg_counts[stem] / total_neg_tokens
            if (p_target / (p_neg + smoothing)) < contrast_ratio:
                continue

        valid_vocab.add(stem)

    return valid_vocab


def count_cooccurrences(
    docs: list[list[str]], valid_vocab: set[str], window_size: int
) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    co_counts: Counter[tuple[str, str]] = Counter()
    degree_k: Counter[str] = Counter()
    total_cooccurrences = 0

    for tokens in docs:
        filtered_tokens = [t for t in tokens if t in valid_vocab]
        num_tokens = len(filtered_tokens)

        for i in range(num_tokens):
            u = filtered_tokens[i]
            limit = min(num_tokens, i + window_size)
            for j in range(i + 1, limit):
                v = filtered_tokens[j]
                if u == v:
                    continue
                edge = (u, v) if u < v else (v, u)
                co_counts[edge] += 1
                degree_k[u] += 1
                degree_k[v] += 1
                total_cooccurrences += 1

    return co_counts, degree_k, total_cooccurrences


def build_ppmi_graph(
    co_counts: Counter[tuple[str, str]],
    degree_k: Counter[str],
    total_cooccurrences: int,
    epsilon: float,
) -> nx.Graph:
    graph = nx.Graph()

    if total_cooccurrences == 0:
        return graph

    for (u, v), c_uv in co_counts.items():
        k_u = degree_k[u]
        k_v = degree_k[v]

        if k_u == 0 or k_v == 0:
            continue

        ratio = (c_uv * total_cooccurrences) / (k_u * k_v)
        w_uv = max(0.0, math.log(ratio)) if ratio > 0 else 0.0

        if w_uv >= epsilon and w_uv > 0.0:
            graph.add_edge(u, v, weight=w_uv, raw_count=c_uv)

    for node in graph.nodes():
        graph.nodes[node]["total_cooccurrence_degree"] = degree_k[node]

    return graph


def export_graph(graph: nx.Graph, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".graphml":
        nx.write_graphml(graph, output_path)
    elif suffix == ".gexf":
        nx.write_gexf(graph, output_path)
    elif suffix == ".gml":
        nx.write_gml(graph, output_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .graphml, .gexf, or .gml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PPMI-weighted word stem graph using an NLTK reference corpus for contrastive filtering."
    )
    parser.add_argument("corpus_dir", type=Path, help="Root folder containing target .md/.txt files")
    parser.add_argument(
        "--nltk-neg",
        type=str,
        default="brown",
        choices=["brown", "reuters", "gutenberg", "none"],
        help="Built-in NLTK background corpus to filter generic words (default: brown)",
    )
    parser.add_argument(
        "-n",
        "--negative-dir",
        type=Path,
        default=None,
        help="Optional custom folder of negative .md/.txt files (overrides --nltk-neg if set)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("corpus_skeleton.graphml"),
        help="Output graph path (.graphml, .gexf, or .gml)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=5,
        help="Sliding window size (default: 5)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1.0,
        help="Minimum PPMI weight threshold epsilon (default: 1.0)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.7,
        help="Drop words appearing in more than this fraction of target documents (default: 0.7)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum total occurrences required to keep a word (default: 2)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=3.0,
        help="Minimum target-to-negative frequency ratio (default: 3.0)",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["english"],
        help="NLTK stopword languages (default: english)",
    )

    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {args.corpus_dir}")

    stemmer = PorterStemmer()
    stop_stems = get_stemmed_stopwords(stemmer, args.lang)

    target_docs = load_corpus_documents(args.corpus_dir, stemmer, stop_stems)

    neg_counts: Counter[str] = Counter()
    total_neg_tokens = 0

    if args.negative_dir is not None:
        if not args.negative_dir.is_dir():
            raise FileNotFoundError(f"Negative directory not found: {args.negative_dir}")
        neg_docs = load_corpus_documents(args.negative_dir, stemmer, stop_stems)
        for doc in neg_docs:
            neg_counts.update(doc)
            total_neg_tokens += len(doc)
    elif args.nltk_neg != "none":
        neg_counts, total_neg_tokens = load_nltk_background_counts(
            args.nltk_neg, stemmer, stop_stems
        )

    valid_vocab = build_valid_vocabulary(
        target_docs=target_docs,
        neg_counts=neg_counts,
        total_neg_tokens=total_neg_tokens,
        max_df_ratio=args.max_df,
        min_count=args.min_count,
        contrast_ratio=args.contrast,
    )

    co_counts, degree_k, total_n = count_cooccurrences(target_docs, valid_vocab, args.window)
    graph = build_ppmi_graph(co_counts, degree_k, total_n, args.epsilon)
    export_graph(graph, args.output)

    print(f"Target documents loaded: {len(target_docs)}")
    print(f"Negative reference tokens: {total_neg_tokens}")
    print(f"Valid vocabulary size: {len(valid_vocab)}")
    print(f"Total co-occurrences (N): {total_n}")
    print(f"Surviving nodes: {graph.number_of_nodes()}")
    print(f"Surviving edges: {graph.number_of_edges()}")
    print(f"Saved graph to: {args.output}")


if __name__ == "__main__":
    main()