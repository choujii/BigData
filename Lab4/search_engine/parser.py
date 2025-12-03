import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

from bs4 import BeautifulSoup

WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Document:
    doc_id: str
    text: str
    words: List[str]
    out_links: List[str]


def parse_html_document(path: Path, url_map: Dict[str, str]) -> Document:
    """
    Парсим реальный HTML:
    - вытаскиваем текст
    - вытаскиваем ссылки <a href="...">
    - оставляем только те ссылки, которые ведут на наши же документы
    """
    doc_id = path.stem
    html = path.read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(separator=" ")

    words = [w.lower() for w in WORD_RE.findall(text)]

    hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]

    reverse_url_map = {v: k for k, v in url_map.items()}

    out_links: List[str] = []
    for href in hrefs:
        if href in reverse_url_map:
            out_links.append(reverse_url_map[href])

    return Document(
        doc_id=doc_id,
        text=text,
        words=words,
        out_links=out_links
    )


def parse_txt_document(path: Path) -> Document:
    """
    Старый вариант для .txt с [link:docX] — можешь оставить,
    если хочешь использовать и текстовые файлы.
    """
    LINK_RE = re.compile(r"\[link:(\w+)\]")
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")

    words = [w.lower() for w in WORD_RE.findall(text)]
    out_links = LINK_RE.findall(text)

    return Document(
        doc_id=doc_id,
        text=text,
        words=words,
        out_links=out_links
    )


def parse_corpus(data_dir: str) -> Dict[str, Document]:
    """
    Читает все .html и .txt из data_dir и возвращает dict doc_id -> Document.
    Для .html используем карту doc_id -> url.
    """
    data_path = Path(data_dir)
    docs: Dict[str, Document] = {}

    url_map = {
        "doc1": "https://ru.wikipedia.org/wiki/Парусный_спорт",
        "doc2": "https://ru.wikipedia.org/wiki/Яхта",
        "doc3": "https://ru.wikipedia.org/wiki/Регата",
        "doc4": "https://ru.wikipedia.org/wiki/Ветер",
    }

    for name in os.listdir(data_path):
        path = data_path / name
        if name.endswith(".html"):
            doc = parse_html_document(path, url_map)
        elif name.endswith(".txt"):
            doc = parse_txt_document(path)
        else:
            continue
        docs[doc.doc_id] = doc

    return docs
