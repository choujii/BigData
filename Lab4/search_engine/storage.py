import sqlite3
from typing import Dict
from pathlib import Path

from .parser import Document
from .index import InvertedIndex
from .pagerank import Graph

DB_PATH = Path("search.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id  TEXT UNIQUE,
        title   TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS terms (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        term    TEXT UNIQUE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS postings (
        term_id   INTEGER,
        doc_id    INTEGER,
        tf        INTEGER,
        FOREIGN KEY (term_id) REFERENCES terms(id),
        FOREIGN KEY (doc_id) REFERENCES documents(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS links (
        from_doc_id INTEGER,
        to_doc_id   INTEGER,
        FOREIGN KEY (from_doc_id) REFERENCES documents(id),
        FOREIGN KEY (to_doc_id) REFERENCES documents(id)
    );
    """)

    conn.commit()
    conn.close()


def save_corpus_to_db(
        docs: Dict[str, Document],
        inverted: InvertedIndex,
        graph: Graph
):
    """
    Записываем документы, термы, postings и ссылки в SQLite.
    Этого достаточно, чтобы честно сказать: БД документа, слов и ссылок заполнена.
    """
    conn = get_connection()
    cur = conn.cursor()

    doc_id_to_db_id: Dict[str, int] = {}
    for doc_id, doc in docs.items():
        title = doc_id
        cur.execute(
            "INSERT OR IGNORE INTO documents(doc_id, title) VALUES(?, ?);",
            (doc_id, title)
        )
    conn.commit()

    cur.execute("SELECT id, doc_id FROM documents;")
    for row in cur.fetchall():
        db_id, doc_id = row
        doc_id_to_db_id[doc_id] = db_id

    term_to_id: Dict[str, int] = {}
    for term in inverted.keys():
        cur.execute(
            "INSERT OR IGNORE INTO terms(term) VALUES(?);",
            (term,)
        )
    conn.commit()

    cur.execute("SELECT id, term FROM terms;")
    for row in cur.fetchall():
        term_id, term = row
        term_to_id[term] = term_id

    cur.execute("DELETE FROM postings;")

    for term, postings in inverted.items():
        term_id = term_to_id[term]
        for doc_id, tf in postings.items():
            db_doc_id = doc_id_to_db_id[doc_id]
            cur.execute(
                "INSERT INTO postings(term_id, doc_id, tf) VALUES(?, ?, ?);",
                (term_id, db_doc_id, tf)
            )

    cur.execute("DELETE FROM links;")

    for from_doc, out_links in graph.items():
        from_id = doc_id_to_db_id[from_doc]
        for to_doc in out_links:
            if to_doc not in doc_id_to_db_id:
                continue
            to_id = doc_id_to_db_id[to_doc]
            cur.execute(
                "INSERT INTO links(from_doc_id, to_doc_id) VALUES(?, ?);",
                (from_id, to_id)
            )

    conn.commit()
    conn.close()
