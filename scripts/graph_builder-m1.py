import os
import sys
import time
from pathlib import Path

# Set environment variable before importing nltk
LOCAL_NLTK_DIR = Path(__file__).resolve().parent.parent / ".nltk_data"
LOCAL_NLTK_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
LOCAL_NLTK_DIR.chmod(0o755)
os.environ["NLTK_DATA"] = str(LOCAL_NLTK_DIR)

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_stemmed_stopwords(stemmer: PorterStemmer, languages: list[str]) -> set[str]:
    log(f"Checking/downloading NLTK stopwords to {LOCAL_NLTK_DIR}...")
    nltk.download("stopwords", download_dir=str(LOCAL_NLTK_DIR), quiet=True)
    raw_stops: set[str] = set()
    for lang in languages:
        raw_stops.update(stopwords.words(lang))
    stemmed_stops = {stemmer.stem(w) for w in raw_stops}
    combined = raw_stops | stemmed_stops
    log(f"Loaded {len(combined)} raw and stemmed stopword entries for: {', '.join(languages)}")
    return combined


def load_nltk_background_counts(
    corpus_name: str,
    stemmer: PorterStemmer,
    stop_stems: set[str],
    languages: list[str],
    use_cache: bool = True,
) -> tuple[Counter[str], int]:
    cache_dir = LOCAL_NLTK_DIR / "cache"
    cache_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    lang_key = "-".join(sorted(set(languages)))
    cache_file = cache_dir / f"{corpus_name}_{lang_key}_porter.json"

    if use_cache and cache_file.is_file():
        log(f"Loading cached '{corpus_name}' background counts from {cache_file}...")
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            neg_counts: Counter[str] = Counter(payload["counts"])
            total_neg_tokens = int(payload["total_neg_tokens"])
            log(
                f"Background cache loaded: {total_neg_tokens} valid tokens, "
                f"{len(neg_counts)} unique stems."
            )
            return neg_counts, total_neg_tokens
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            log(f"Cache file unreadable ({exc}); rebuilding...")

    log(f"Checking/downloading NLTK '{corpus_name}' background corpus...")
    nltk.download(corpus_name, download_dir=str(LOCAL_NLTK_DIR), quiet=True)
    corpus_module = getattr(nltk.corpus, corpus_name)

    raw_words = corpus_module.words()
    total_raw = len(raw_words)
    log(f"Stemming {total_raw} words from NLTK '{corpus_name}' corpus...")

    neg_counts = Counter()
    total_neg_tokens = 0

    step = max(1, total_raw // 10)
    for idx, w in enumerate(raw_words, 1):
        if idx % step == 0 or idx == total_raw:
            pct = (idx / total_raw) * 100.0
            print(f"\r  -> Background progress: {idx}/{total_raw} ({pct:.1f}%)", end="", flush=True)

        w_low = w.lower()
        if len(w_low) < 2 or not w_low.isalpha() or w_low in stop_stems:
            continue
        stem = stemmer.stem(w_low)
        if stem not in stop_stems:
            neg_counts[stem] += 1
            total_neg_tokens += 1

    print(flush=True)

    payload = {
        "corpus": corpus_name,
        "languages": sorted(set(languages)),
        "total_neg_tokens": total_neg_tokens,
        "counts": dict(neg_counts),
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    log(
        f"Background corpus ready and cached to {cache_file}: "
        f"{total_neg_tokens} valid tokens, {len(neg_counts)} unique stems."
    )
    return neg_counts, total_neg_tokens


def extract_and_stem_tokens(
    file_path: Path,
    stemmer: PorterStemmer,
    stop_stems: set[str],
    stem_surface_counts: dict[str, Counter[str]] | None = None,
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
            if stem_surface_counts is not None:
                stem_surface_counts[stem][w] += 1
    return tokens


def load_corpus_documents(
    corpus_dir: Path,
    stemmer: PorterStemmer,
    stop_stems: set[str],
    label: str = "target",
    track_surface: bool = False,
) -> tuple[list[list[str]], dict[str, Counter[str]]]:
    log(f"Scanning {corpus_dir} for {label} files (.md, .txt)...")
    files = [
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    ]
    total_files = len(files)
    log(f"Found {total_files} {label} files. Extracting and stemming tokens...")

    docs: list[list[str]] = []
    surface_map: dict[str, Counter[str]] = defaultdict(Counter)
    total_tokens = 0
    empty_files = 0

    step = max(1, total_files // 20) if total_files > 0 else 1
    for idx, file_path in enumerate(files, 1):
        tokens = extract_and_stem_tokens(
            file_path,
            stemmer,
            stop_stems,
            stem_surface_counts=surface_map if track_surface else None,
        )
        if tokens:
            docs.append(tokens)
            total_tokens += len(tokens)
        else:
            empty_files += 1

        if idx % step == 0 or idx == total_files:
            pct = (idx / total_files) * 100.0
            print(
                f"\r  -> Processed {idx}/{total_files} files ({pct:.1f}%) | "
                f"Tokens: {total_tokens}",
                end="",
                flush=True,
            )

    if total_files > 0:
        print(flush=True)

    log(
        f"Finished loading {label} corpus: {len(docs)} non-empty documents "
        f"({empty_files} empty/skipped), {total_tokens} total stemmed tokens."
    )
    return docs, surface_map


def resolve_canonical_words(
    surface_map: dict[str, Counter[str]], valid_vocab: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    canonical_map: dict[str, str] = {}
    variants_map: dict[str, str] = {}

    for stem in valid_vocab:
        counts = surface_map.get(stem)
        if not counts:
            canonical_map[stem] = stem
            variants_map[stem] = stem
            continue

        ranked = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        canonical_map[stem] = ranked[0][0]
        variants_map[stem] = ", ".join(f"{w}({c})" for w, c in ranked[:5])

    return canonical_map, variants_map


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

    log("Building vocabulary and applying frequency/contrast filters...")
    target_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    total_target_tokens = 0

    for doc in target_docs:
        target_counts.update(doc)
        doc_freq.update(set(doc))
        total_target_tokens += len(doc)

    valid_vocab: set[str] = set()
    smoothing = 1e-6

    dropped_min_count = 0
    dropped_max_df = 0
    dropped_contrast = 0

    for stem, count in target_counts.items():
        if count < min_count:
            dropped_min_count += 1
            continue

        if num_docs > 1 and (doc_freq[stem] / num_docs) > max_df_ratio:
            dropped_max_df += 1
            continue

        if total_neg_tokens > 0:
            p_target = count / total_target_tokens
            p_neg = neg_counts[stem] / total_neg_tokens
            if (p_target / (p_neg + smoothing)) < contrast_ratio:
                dropped_contrast += 1
                continue

        valid_vocab.add(stem)

    log(
        f"Vocabulary filtering breakdown (from {len(target_counts)} unique candidate stems):\n"
        f"  - Dropped by min_count (< {min_count}): {dropped_min_count}\n"
        f"  - Dropped by max_df (> {max_df_ratio:.2f}): {dropped_max_df}\n"
        f"  - Dropped by negative contrast (< {contrast_ratio:.2f}): {dropped_contrast}\n"
        f"  - Surviving vocabulary size: {len(valid_vocab)}"
    )
    return valid_vocab


def count_cooccurrences(
    docs: list[list[str]], valid_vocab: set[str], window_size: int
) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    log(f"Counting co-occurrences with sliding window size = {window_size}...")
    co_counts: Counter[tuple[str, str]] = Counter()
    degree_k: Counter[str] = Counter()
    total_cooccurrences = 0

    total_docs = len(docs)
    step = max(1, total_docs // 20) if total_docs > 0 else 1

    for doc_idx, tokens in enumerate(docs, 1):
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

        if doc_idx % step == 0 or doc_idx == total_docs:
            pct = (doc_idx / total_docs) * 100.0
            print(
                f"\r  -> Window scan: {doc_idx}/{total_docs} docs ({pct:.1f}%) | "
                f"Unique pairs: {len(co_counts)} | Total N: {total_cooccurrences}",
                end="",
                flush=True,
            )

    if total_docs > 0:
        print(flush=True)

    log(
        f"Co-occurrence counting complete: {len(co_counts)} unique candidate edges, "
        f"{total_cooccurrences} total co-occurrences (N)."
    )
    return co_counts, degree_k, total_cooccurrences


def build_ppmi_graph(
    co_counts: Counter[tuple[str, str]],
    degree_k: Counter[str],
    total_cooccurrences: int,
    min_cooc: int,
    epsilon: float,
    hub_penalty_alpha: float,
    min_pull: float,
    top_k: int,
    canonical_map: dict[str, str],
    variants_map: dict[str, str],
) -> nx.Graph:
    log(
        f"Stage 1 pruning: filtering edges with raw_count >= {min_cooc} "
        f"and PPMI >= {epsilon}..."
    )
    stem_graph = nx.Graph()

    if total_cooccurrences == 0:
        log("No co-occurrences found. Returning empty graph.")
        return stem_graph

    dropped_min_cooc = 0
    dropped_ppmi = 0

    for (u, v), c_uv in co_counts.items():
        if c_uv < min_cooc:
            dropped_min_cooc += 1
            continue

        k_u = degree_k[u]
        k_v = degree_k[v]
        if k_u == 0 or k_v == 0:
            continue

        ratio = (c_uv * total_cooccurrences) / (k_u * k_v)
        ppmi = max(0.0, math.log(ratio)) if ratio > 0 else 0.0

        if ppmi >= epsilon and ppmi > 0.0:
            base_weight = float(c_uv) * ppmi
            stem_graph.add_edge(
                u,
                v,
                raw_count=int(c_uv),
                ppmi=float(ppmi),
                base_weight=base_weight,
            )
        else:
            dropped_ppmi += 1

    log(
        f"Stage 1 results:\n"
        f"  - Candidate pairs: {len(co_counts)}\n"
        f"  - Dropped by min_cooc (< {min_cooc}): {dropped_min_cooc}\n"
        f"  - Dropped by PPMI (< {epsilon}): {dropped_ppmi}\n"
        f"  - Edges entering Stage 2: {stem_graph.number_of_edges()}"
    )

    if stem_graph.number_of_edges() == 0:
        return stem_graph

    log(
        f"Stage 2 weighting: computing bidirectional hub-penalized pull "
        f"(alpha = {hub_penalty_alpha}, min_pull = {min_pull}, top_k = {top_k})..."
    )

    node_strength: dict[str, float] = {}
    node_deg: dict[str, int] = {}
    for node in stem_graph.nodes():
        incident_edges = stem_graph[node]
        node_deg[node] = len(incident_edges)
        node_strength[node] = sum(data["base_weight"] for data in incident_edges.values())

    edges_to_remove: list[tuple[str, str]] = []
    for u, v, data in stem_graph.edges(data=True):
        b_uv = data["base_weight"]
        s_u = node_strength[u]
        s_v = node_strength[v]
        d_u = node_deg[u]
        d_v = node_deg[v]

        denom = math.sqrt(s_u * s_v) * ((d_u * d_v) ** (hub_penalty_alpha / 2.0))
        w_pull = (b_uv / denom) if denom > 0 else 0.0

        if w_pull < min_pull:
            edges_to_remove.append((u, v))
        else:
            data["weight"] = float(w_pull)
            data["pull_weight"] = float(w_pull)

    dropped_min_pull = len(edges_to_remove)
    stem_graph.remove_edges_from(edges_to_remove)

    dropped_top_k = 0
    if top_k > 0 and stem_graph.number_of_edges() > 0:
        allowed_by_node: dict[str, set[str]] = {}
        for node in stem_graph.nodes():
            nbrs = sorted(
                stem_graph[node].items(),
                key=lambda item: item[1]["weight"],
                reverse=True,
            )
            allowed_by_node[node] = {nbr for nbr, _ in nbrs[:top_k]}

        k_remove: list[tuple[str, str]] = []
        for u, v in stem_graph.edges():
            if v not in allowed_by_node[u] or u not in allowed_by_node[v]:
                k_remove.append((u, v))

        dropped_top_k = len(k_remove)
        stem_graph.remove_edges_from(k_remove)

    isolates = list(nx.isolates(stem_graph))
    stem_graph.remove_nodes_from(isolates)

    relabel_mapping: dict[str, str] = {}
    used_names: Counter[str] = Counter()

    for stem in stem_graph.nodes():
        word = canonical_map.get(stem, stem)
        used_names[word] += 1
        node_id = word if used_names[word] == 1 else f"{word}_{stem}"
        relabel_mapping[stem] = node_id

        stem_graph.nodes[stem]["label"] = word
        stem_graph.nodes[stem]["stem"] = stem
        stem_graph.nodes[stem]["variants"] = variants_map.get(stem, word)
        stem_graph.nodes[stem]["total_cooccurrence_degree"] = int(degree_k[stem])
        stem_graph.nodes[stem]["skeleton_degree"] = int(stem_graph.degree[stem])

    graph = nx.relabel_nodes(stem_graph, relabel_mapping, copy=True)

    surviving_weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    if surviving_weights:
        min_w = min(surviving_weights)
        max_w = max(surviving_weights)
        avg_w = sum(surviving_weights) / len(surviving_weights)
        weight_stats = f"min={min_w:.4f}, avg={avg_w:.4f}, max={max_w:.4f}"
    else:
        weight_stats = "n/a"

    num_components = nx.number_connected_components(graph) if graph.number_of_nodes() > 0 else 0

    log(
        f"Stage 2 pruning and canonical unstemming complete:\n"
        f"  - Dropped by min_pull (< {min_pull}): {dropped_min_pull}\n"
        f"  - Dropped by mutual top_k (> {top_k}): {dropped_top_k}\n"
        f"  - Isolated nodes removed: {len(isolates)}\n"
        f"  - Final surviving edges: {graph.number_of_edges()} (pull weight {weight_stats})\n"
        f"  - Final surviving nodes: {graph.number_of_nodes()}\n"
        f"  - Connected components: {num_components}"
    )

    if graph.number_of_nodes() > 0:
        top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:10]
        top_str = ", ".join(
            f"{n} [stem: {graph.nodes[n]['stem']}] ({d})" for n, d in top_nodes
        )
        log(f"Top 10 nodes by final skeleton degree: {top_str}")

        top_edges = sorted(
            graph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
        )[:10]
        edge_str = ", ".join(
            f"{u}<->{v} (pull={d['weight']:.3f}, cooc={d['raw_count']})"
            for u, v, d in top_edges
        )
        log(f"Top 10 strongest bidirectional links: {edge_str}")

    return graph


def export_graph(graph: nx.Graph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    log(f"Exporting graph to {output_path} (format: {suffix})...")
    if suffix == ".graphml":
        nx.write_graphml(graph, output_path)
    elif suffix == ".gexf":
        nx.write_gexf(graph, output_path)
    elif suffix == ".gml":
        nx.write_gml(graph, output_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .graphml, .gexf, or .gml")
    log(f"Export complete: {output_path.resolve()}")


def main() -> None:
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Build a PPMI and bidirectional degree-penalized word graph from a Markdown corpus."
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
        "--no-cache",
        action="store_true",
        help="Bypass and rebuild the local stemmed cache for the NLTK background corpus",
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
        "--min-cooc",
        type=int,
        default=2,
        help="Minimum raw co-occurrences C(u,v) required for an edge (default: 2)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=2.0,
        help="Minimum PPMI weight threshold epsilon (default: 2.0)",
    )
    parser.add_argument(
        "--hub-penalty",
        type=float,
        default=0.5,
        help="Exponent alpha penalizing high-degree nodes in bidirectional pull (default: 0.5)",
    )
    parser.add_argument(
        "--min-pull",
        type=float,
        default=0.02,
        help="Minimum bidirectional pull weight in (0, 1] required to survive (default: 0.02)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=15,
        help="Keep at most top K strongest bidirectional links per node; 0 to disable (default: 15)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.5,
        help="Drop words appearing in more than this fraction of target documents (default: 0.5)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum total occurrences required to keep a word (default: 3)",
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

    log(f"Starting graph_builder-m1 | Local NLTK directory: {LOCAL_NLTK_DIR}")
    stemmer = PorterStemmer()
    stop_stems = get_stemmed_stopwords(stemmer, args.lang)

    target_docs, surface_map = load_corpus_documents(
        args.corpus_dir, stemmer, stop_stems, label="target", track_surface=True
    )
    if not target_docs:
        log("No valid tokens found in target corpus. Exiting.")
        sys.exit(0)

    neg_counts: Counter[str] = Counter()
    total_neg_tokens = 0

    if args.negative_dir is not None:
        if not args.negative_dir.is_dir():
            raise FileNotFoundError(f"Negative directory not found: {args.negative_dir}")
        neg_docs, _ = load_corpus_documents(
            args.negative_dir, stemmer, stop_stems, label="negative", track_surface=False
        )
        for doc in neg_docs:
            neg_counts.update(doc)
            total_neg_tokens += len(doc)
    elif args.nltk_neg != "none":
        neg_counts, total_neg_tokens = load_nltk_background_counts(
            corpus_name=args.nltk_neg,
            stemmer=stemmer,
            stop_stems=stop_stems,
            languages=args.lang,
            use_cache=not args.no_cache,
        )
    else:
        log("Negative corpus filtering disabled (--nltk-neg none).")

    valid_vocab = build_valid_vocabulary(
        target_docs=target_docs,
        neg_counts=neg_counts,
        total_neg_tokens=total_neg_tokens,
        max_df_ratio=args.max_df,
        min_count=args.min_count,
        contrast_ratio=args.contrast,
    )

    canonical_map, variants_map = resolve_canonical_words(surface_map, valid_vocab)
    co_counts, degree_k, total_n = count_cooccurrences(target_docs, valid_vocab, args.window)
    graph = build_ppmi_graph(
        co_counts=co_counts,
        degree_k=degree_k,
        total_cooccurrences=total_n,
        min_cooc=args.min_cooc,
        epsilon=args.epsilon,
        hub_penalty_alpha=args.hub_penalty,
        min_pull=args.min_pull,
        top_k=args.top_k,
        canonical_map=canonical_map,
        variants_map=variants_map,
    )
    export_graph(graph, args.output)

    elapsed = time.monotonic() - start_time
    log(f"Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()