from typing import Dict, List, Tuple, Callable, Any, DefaultDict
from collections import defaultdict

from .parser import Document

Graph = Dict[str, List[str]]
Ranks = Dict[str, float]


def build_graph(docs: Dict[str, Document]) -> Graph:
    graph: Graph = {}
    for doc_id, doc in docs.items():
        graph[doc_id] = [dst for dst in doc.out_links if dst in docs]
    return graph


# =========================
#  PageRank: MapReduce-style
# =========================

def pagerank_mapreduce(
        graph: Graph,
        num_iters: int = 10,
        d: float = 0.85
) -> Ranks:
    """
    Реализация PageRank в стиле MapReduce (map + group + reduce).
    В реальном MapReduce это было бы в кластере, здесь — в памяти.
    """
    nodes = list(graph.keys())
    n = len(nodes)
    ranks: Ranks = {v: 1.0 / n for v in nodes}

    for _ in range(num_iters):
        contributions: DefaultDict[str, float] = defaultdict(float)

        dangling_sum = 0.0

        for v in nodes:
            out_links = graph[v]
            rank_v = ranks[v]

            if not out_links:
                dangling_sum += rank_v
                continue

            contrib = rank_v / len(out_links)
            for dst in out_links:
                contributions[dst] += contrib

        dangling_contrib = dangling_sum / n if n > 0 else 0.0

        new_ranks: Ranks = {}
        for v in nodes:
            sum_in = contributions[v] + dangling_contrib
            new_ranks[v] = (1 - d) / n + d * sum_in

        ranks = new_ranks

    return ranks


# =========================
#  Pregel-like PageRank
# =========================

VertexState = float
Message = float

VertexProgram = Callable[[str, VertexState, Message], VertexState]
SendMessage = Callable[[str, VertexState, List[str]], List[Tuple[str, Message]]]
MergeMessage = Callable[[Message, Message], Message]


def run_pregel(
        graph: Graph,
        initial_state: VertexState,
        vprog: VertexProgram,
        send_msg: SendMessage,
        merge_msg: MergeMessage,
        num_iters: int
) -> Dict[str, VertexState]:
    """
    Мини-реализация Pregel-модели:
    - на каждой итерации у вершины есть state и входящее сообщение msg
    - vprog обновляет состояние
    - send_msg генерирует сообщения соседям
    - merge_msg объединяет сообщения, пришедшие к одной вершине
    """
    nodes = list(graph.keys())
    state: Dict[str, VertexState] = {v: initial_state for v in nodes}
    messages: Dict[str, Message] = {v: 0.0 for v in nodes}

    for _ in range(num_iters):
        new_state: Dict[str, VertexState] = {}
        for v in nodes:
            new_state[v] = vprog(v, state[v], messages.get(v, 0.0))

        state = new_state

        new_messages: DefaultDict[str, Message] = defaultdict(float)
        for v in nodes:
            outgoing = send_msg(v, state[v], graph[v])
            for dst, msg in outgoing:
                if dst not in state:
                    continue
                if dst in new_messages:
                    new_messages[dst] = merge_msg(new_messages[dst], msg)
                else:
                    new_messages[dst] = msg

        messages = {v: new_messages.get(v, 0.0) for v in nodes}

    return state


def pagerank_pregel(
        graph: Graph,
        num_iters: int = 10,
        d: float = 0.85
) -> Ranks:
    """
    PageRank поверх нашей mini-Pregel-модели.
    """
    nodes = list(graph.keys())
    n = len(nodes)
    initial_rank = 1.0 / n if n > 0 else 0.0

    def vprog(v: str, old_rank: float, msg_sum: float) -> float:
        return (1 - d) / n + d * msg_sum

    def send_msg(v: str, rank: float, out_neighbors: List[str]):
        if not out_neighbors:
            return []
        contrib = rank / len(out_neighbors)
        return [(dst, contrib) for dst in out_neighbors]

    def merge_msg(a: float, b: float) -> float:
        return a + b

    result = run_pregel(
        graph=graph,
        initial_state=initial_rank,
        vprog=vprog,
        send_msg=send_msg,
        merge_msg=merge_msg,
        num_iters=num_iters
    )

    return result
