from typing import Dict

from .parser import parse_corpus, Document
from .index import build_inverted_index, compute_idf, pretty_print_index
from .pagerank import build_graph, pagerank_mapreduce, pagerank_pregel
from .search import taat_search, daat_search, apply_pagerank_boost
from .storage import init_db, save_corpus_to_db


def run_demo():
    print("=== Мини-поисковик (ЛР4) ===")

    # 1. Парсим коллекцию документов
    data_dir = "data"
    docs: Dict[str, Document] = parse_corpus(data_dir)
    print(f"Загружено документов: {len(docs)}")
    print("Документы:", ", ".join(sorted(docs.keys())))
    print()

    # 2. Строим инвертированный индекс
    inverted = build_inverted_index(docs)
    num_docs = len(docs)
    idf = compute_idf(inverted, num_docs)

    print("Покажем кусок инвертированного индекса:")
    pretty_print_index({k: inverted[k] for k in list(inverted.keys())[:10]})
    print()

    # 3. Строим граф ссылок и считаем PageRank (MapReduce-style)
    graph = build_graph(docs)
    print("Инициализирую и заполняю базу данных SQLite (search.db)...")
    init_db()
    save_corpus_to_db(docs, inverted, graph)
    print("База данных заполнена.\n")

    pr_mr = pagerank_mapreduce(graph, num_iters=10, d=0.85)
    print("PageRank (MapReduce-стиль):")
    for doc_id, rank in sorted(pr_mr.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_id}: {rank:.4f}")
    print()

    # 4. PageRank с использованием Pregel-подобной модели
    pr_pregel = pagerank_pregel(graph, num_iters=10, d=0.85)
    print("PageRank (Pregel-модель):")
    for doc_id, rank in sorted(pr_pregel.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_id}: {rank:.4f}")
    print()

    # 5. Поиск
    print("Теперь можно ввести поисковый запрос.")
    print("Примеры: 'парусный спорт', 'регата', 'яхты скорость'")
    try:
        query = input("Введите запрос: ").strip()
    except EOFError:
        query = "парусный спорт"

    if not query:
        query = "парусный спорт"

    print(f"\nЗапрос: {query!r}\n")

    # TAAT
    print("=== Поиск (Term-at-a-time, TF-IDF) ===")
    taat_results = taat_search(query, inverted, idf, num_docs)
    for doc_id, score in taat_results[:10]:
        print(f"  {doc_id}: score={score:.4f}")
    print()

    # DAAT
    print("=== Поиск (Document-at-a-time, TF-IDF) ===")
    daat_results = daat_search(query, inverted, idf, num_docs)
    for doc_id, score in daat_results[:10]:
        print(f"  {doc_id}: score={score:.4f}")
    print()

    # Комбинация с PageRank (MapReduce)
    print("=== Поиск (TAAT + PageRank MapReduce, комбинированный скор) ===")
    boosted_results = apply_pagerank_boost(taat_results, pr_mr, alpha=0.8)
    for doc_id, score in boosted_results[:10]:
        print(f"  {doc_id}: score={score:.4f}, PR={pr_mr.get(doc_id, 0):.4f}")
    print()

    print("Демо завершено.")
