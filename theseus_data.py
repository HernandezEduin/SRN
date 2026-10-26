"""Load THESEUS KGQA releases without changing SRN's answer-only supervision.

The annotated gold paths are deliberately not read by this adapter.
"""

import ast
import csv
import re
from pathlib import Path
from urllib.request import urlopen

import networkx as nx


RELEASES = {
    "kinship": "HalcyonSolutions/Kinship",
    "mquake-st": "HalcyonSolutions/MQuAKE-ST",
}


def release_paths(dataset, hops="2", answer_mode="single", kinship_graph="orig"):
    if dataset == "kinship":
        if hops not in ("2", "3", "n"):
            raise ValueError("Kinship supports 2, 3, and n hops")
        if kinship_graph != "orig":
            raise ValueError("Use kg/orig for navigation. The gender graph is KGE-only")
        return "kg/orig/triplets.txt", "qa/kinship_qa_{}hop.csv".format(hops)
    if dataset == "mquake-st":
        if hops not in ("2", "3", "4", "n"):
            raise ValueError("MQuAKE-ST supports 2, 3, 4, and n hops")
        if answer_mode not in ("single", "multi"):
            raise ValueError("answer_mode must be single or multi")
        answers = "single_answers" if answer_mode == "single" else "multi_answers"
        return "kg/triplets.txt", "qa/{}/qa_{}hop.csv".format(answers, hops)
    raise ValueError("Unknown dataset: {}".format(dataset))


def download_file(dataset, remote_path, data_root):
    """Cache an official release file, keeping existing files untouched."""
    path = Path(data_root) / dataset / remote_path
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/{}/resolve/main/{}".format(
        RELEASES[dataset], remote_path
    )
    temp_path = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=120) as source, temp_path.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def load_graph(path):
    """Directed, edge-labelled MultiDiGraph. Parallel relations are preserved."""
    kg = nx.MultiDiGraph()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            triple = line.rstrip("\r\n").split("\t")
            if len(triple) != 3:
                raise ValueError("{}:{} expected three tab-separated fields".format(path, lineno))
            head, relation, tail = triple
            kg.add_edge(head, tail, relation=relation)
    return kg


def parse_answers(value):
    """Single-answer scalar or multi-answer Python-literal list from HF CSV."""
    value = value.strip()
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (tuple, list)):
            raise ValueError("Invalid multi-answer field")
        return frozenset(str(answer) for answer in parsed)
    return frozenset((value,))


def question_tokens(question, source_label):
    """Use explicit topic labels and preserve SRN's <e> entity placeholder."""
    # Entity names in MQuAKE-ST may contain whitespace and punctuation.
    question = re.sub(re.escape(source_label), " <e> ", question, flags=re.IGNORECASE)
    tokens = re.findall(r"<e>|[\w]+(?:'[\w]+)?", question.lower(), re.UNICODE)
    return tokens or ["<e>"]


def load_questions(path, graph, dataset):
    """Return split->list of (tokens, topic, answer_set, hop_count, question_id)."""
    splits = {"train": [], "dev": [], "test": []}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            split = row["SplitLabel"].strip().lower()
            if split not in splits:
                raise ValueError("Unexpected split {}".format(split))
            topic = row["Source-Entity"].strip()
            answers = parse_answers(row["Answer-Entity"])
            if not graph.has_node(topic):
                raise ValueError("Question topic {} absent from navigation KG".format(topic))
            if not answers or not answers.issubset(graph.nodes):
                raise ValueError("Question {} has an answer absent from KG".format(row["Question-Number"]))
            tokens = question_tokens(row["Question"], row["Source"])
            splits[split].append((tokens, topic, answers, int(row["Hops"]), row["Question-Number"]))
    return splits


def load_release(dataset, hops="2", answer_mode="single", data_root="datasets/theseus"):
    kg_file, qa_file = release_paths(dataset, hops, answer_mode)
    graph_path = download_file(dataset, kg_file, data_root)
    questions_path = download_file(dataset, qa_file, data_root)
    graph = load_graph(graph_path)
    splits = load_questions(questions_path, graph, dataset)
    return graph, splits, graph_path
