import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#настройки, которые можно крутить и смотреть, как меняется выдача
MIN_RANK = 50            # теги со relevance ниже этого значения выбрасываю
W_TAGS, W_DESC = 0.85, 0.15  # вклад тегов/жанров и описания сюжета в итоговое сходство

#1. загрузка
df = pd.read_csv('data/anilist_anime.csv').reset_index(drop=True)
tags = pd.read_csv('data/anilist_tags.csv')

#2. теги: без спойлеров, без слабых, с весом = rank/100
tags = tags[(~tags['is_spoiler']) & (tags['rank'] >= MIN_RANK)].copy()
tags['weight'] = tags['rank'] / 100
tag_matrix = tags.pivot_table(
    index='id', columns='tag', values='weight', aggfunc='max', fill_value=0
)

#3. жанры: вес 1, префикс, чтобы не путать с тегами
genre_rows = (
    df[['id', 'genres']].dropna()
    .assign(genre=lambda d: d['genres'].str.split(';'))
    .explode('genre')
)
genre_rows['genre'] = 'genre:' + genre_rows['genre']
genre_rows['weight'] = 1.0
genre_matrix = genre_rows.pivot_table(
    index='id', columns='genre', values='weight', aggfunc='max', fill_value=0
)

# строки в том же порядке, что и в df (важно: индекс матрицы = индекс df)
features = pd.concat([tag_matrix, genre_matrix], axis=1).reindex(df['id']).fillna(0)

# TF-IDF поверх весов: тут редкие признаки важнее частых
X_tags = TfidfTransformer().fit_transform(features.values)

#4. описание сюжета
def clean_desc(text):
    text = re.sub(r'\((Source|Written by)[^)]*\)', '', str(text))
    text = re.sub(r'Note:.*', '', text)
    return text

X_desc = TfidfVectorizer(
    stop_words='english', min_df=3, max_df=0.5,
    ngram_range=(1, 2), max_features=20000,
).fit_transform(df['description'].fillna('').map(clean_desc))

print('теги+жанры:', X_tags.shape, '| описание:', X_desc.shape)

#5. итоговое сходство: смесь двух матриц
sim_tags = cosine_similarity(X_tags).astype('float32')
sim_desc = cosine_similarity(X_desc).astype('float32')
sim = W_TAGS * sim_tags + W_DESC * sim_desc
print('sim:', sim.shape)


def franchise_key(title):
    return re.sub(r'[^a-z0-9]', '', title.lower())[:8]


def get_recommendations(title, n=10):
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df['title'].str.lower().str.contains(title.lower(), regex=False)]
    if matches.empty:
        return None, []

    idx = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)

    seen = {franchise_key(df.loc[idx, 'title'])}
    result = []
    for i, score in scores:
        if i == idx:
            continue
        key = franchise_key(df.loc[i, 'title'])
        if key in seen:
            continue
        seen.add(key)
        result.append((df.loc[i, 'title'], float(score)))
        if len(result) == n:
            break

    return df.loc[idx, 'title'], result


if __name__ == '__main__':
    for name in ['Death Note', 'Steins;Gate']:
        found, recs = get_recommendations(name)
        print(f'\nПохожие на: {found}')
        for t, s in recs:
            print(f'  {t}  ({s:.2f})')