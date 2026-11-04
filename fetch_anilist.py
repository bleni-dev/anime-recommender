"""
Скачивает данные с AniList (GraphQL API) и сохраняет в три CSV:

  data/anilist_anime.csv  - одна строка на аниме (название, описание, жанры, метаданные)
  data/anilist_tags.csv   - теги в "длинном" формате: id, tag, rank, is_spoiler
  data/anilist_recs.csv   - фанатские рекомендации: source_id, target_id, rating

Сырые ответы сохраняются в data/anilist_raw.jsonl, поэтому если скрипт
прервётся (лимиты, интернет), при повторном запуске он продолжит с того же места.

Перед запуском:  pip install requests pandas
"""

import html
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://graphql.anilist.co"
PER_PAGE = 50          # максимум для AniList; если ругается на сложность запроса, поставь 25
MAX_ITEMS = 5000       # сколько самых популярных аниме брать
PAUSE = 2.0            # пауза между запросами, сек (лимит API около 90/мин, иногда ниже)

DATA = Path("data")
DATA.mkdir(exist_ok=True)
RAW = DATA / "anilist_raw.jsonl"

QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, format_in: [TV, MOVIE, OVA, ONA], isAdult: false, sort: POPULARITY_DESC) {
      id
      title { romaji english }
      description(asHtml: false)
      genres
      tags { name rank isMediaSpoiler isGeneralSpoiler }
      format
      episodes
      seasonYear
      popularity
      averageScore
      coverImage { large }
      siteUrl
      recommendations(sort: RATING_DESC, perPage: 10) {
        nodes { rating mediaRecommendation { id } }
      }
    }
  }
}
"""


def fetch_page(page):
    """Один запрос к API с повторами при ошибках и лимитах."""
    payload = {"query": QUERY, "variables": {"page": page, "perPage": PER_PAGE}}
    for attempt in range(6):
        try:
            r = requests.post(URL, json=payload, timeout=30)
        except requests.RequestException as e:
            print(f"  сеть: {e}, повтор через 5 сек")
            time.sleep(5)
            continue

        if r.status_code == 429:  # превысили лимит запросов
            wait = int(r.headers.get("Retry-After", 60)) + 1
            print(f"  лимит API, жду {wait} сек")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            print(f"  ошибка сервера {r.status_code}, повтор через 10 сек")
            time.sleep(10)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()
        if "errors" in data:
            raise RuntimeError(data["errors"])
        return data["data"]["Page"]

    raise RuntimeError(f"Не удалось получить страницу {page}")


def download():
    """Качает страницы и дописывает их в jsonl. Продолжает с места остановки."""
    done = 0
    if RAW.exists():
        with open(RAW, encoding="utf-8") as f:
            done = sum(1 for _ in f)
    print(f"Уже скачано: {done}")

    page = done // PER_PAGE + 1
    with open(RAW, "a", encoding="utf-8") as f:
        while done < MAX_ITEMS:
            result = fetch_page(page)
            for media in result["media"]:
                f.write(json.dumps(media, ensure_ascii=False) + "\n")
            f.flush()

            done += len(result["media"])
            print(f"Страница {page}: всего {done}")

            if not result["pageInfo"]["hasNextPage"]:
                break
            page += 1
            time.sleep(PAUSE)


def clean_description(text):
    """Убирает HTML-теги (<br>, <i> и т.п.) и спецсимволы из описания."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def build_tables():
    """Превращает сырой jsonl в три аккуратные таблицы."""
    anime_rows, tag_rows, rec_rows = [], [], []

    with open(RAW, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            aid = m["id"]
            title = m["title"]["romaji"] or m["title"]["english"]

            anime_rows.append({
                "id": aid,
                "title": title,
                "title_english": m["title"]["english"],
                "description": clean_description(m["description"]),
                "genres": ";".join(m["genres"] or []),
                "format": m["format"],
                "episodes": m["episodes"],
                "year": m["seasonYear"],
                "popularity": m["popularity"],
                "score": m["averageScore"],
                "cover": (m["coverImage"] or {}).get("large"),
                "url": m["siteUrl"],
            })

            for t in m["tags"] or []:
                tag_rows.append({
                    "id": aid,
                    "tag": t["name"],
                    "rank": t["rank"],
                    "is_spoiler": bool(t["isMediaSpoiler"] or t["isGeneralSpoiler"]),
                })

            for node in (m["recommendations"] or {}).get("nodes", []):
                target = node.get("mediaRecommendation")
                if target is None:  # рекомендованное аниме могло быть удалено
                    continue
                rec_rows.append({
                    "source_id": aid,
                    "target_id": target["id"],
                    "rating": node["rating"],
                })

    anime = pd.DataFrame(anime_rows).drop_duplicates("id")
    tags = pd.DataFrame(tag_rows)
    recs = pd.DataFrame(rec_rows)

    anime.to_csv(DATA / "anilist_anime.csv", index=False)
    tags.to_csv(DATA / "anilist_tags.csv", index=False)
    recs.to_csv(DATA / "anilist_recs.csv", index=False)

    print(f"\nАниме: {len(anime)}, тегов: {len(tags)}, рекомендаций: {len(recs)}")


if __name__ == "__main__":
    download()
    build_tables()