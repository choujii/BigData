import os
import pathlib
import requests

SITES = {
    "doc1": "https://ru.wikipedia.org/wiki/Парусный_спорт",
    "doc2": "https://ru.wikipedia.org/wiki/Яхта",
    "doc3": "https://ru.wikipedia.org/wiki/Регата",
    "doc4": "https://ru.wikipedia.org/wiki/Ветер",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}


def main():
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)

    for doc_id, url in SITES.items():
        print(f"Скачиваю {url} -> data/{doc_id}.html")

        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()

        (data_dir / f"{doc_id}.html").write_text(resp.text, encoding="utf-8")

    print("Готово: HTML сохранены в папке data/")


if __name__ == "__main__":
    main()
