import numpy as np
import pandas as pd
from scipy import sparse

# переиспользуем таблицу и матрицы сходства из model3_2 (теги+жанры и описание)
from model3_2 import df, sim_tags, sim_desc, franchise_key

#настройки (лучшие значения подберу через evaluate_hybrid.py)
W_DESC = 0.5   # вес описания внутри content-части (лучшее из прошлого теста: 0.5-0.65)
LAM = 0.4     # насколько сильно фанатский граф влияет на итог (0 = только content)
ALPHA = 1.0    # вес связей "через одного посредника" (0 = только прямые рекомендации)

#1. content-сходство
content = ((1 - W_DESC) * sim_tags + W_DESC * sim_desc).astype('float32')

#2. фанатские рекомендации список связей между аним
id2idx = {aid: k for k, aid in enumerate(df['id'])}

recs = pd.read_csv('data/anilist_recs.csv')
recs = recs[recs['rating'] > 0].copy()          # отрицательные "не рекомендую" выбрасываем
recs['a'] = recs['source_id'].map(id2idx)
recs['b'] = recs['target_id'].map(id2idx)
recs = recs.dropna(subset=['a', 'b'])           # рекомендованного аниме нет в моем датасете
recs[['a', 'b']] = recs[['a', 'b']].astype(int)

# связь A-B и B-A считаем одной: упорядочиваем пару и суммируем рейтинги
lo = recs[['a', 'b']].min(axis=1)
hi = recs[['a', 'b']].max(axis=1)
edges = (
    recs.assign(a=lo, b=hi)
    .loc[lambda d: d['a'] != d['b']]
    .groupby(['a', 'b'], as_index=False)['rating'].sum()
)
edges['w'] = np.log1p(edges['rating'])          # log, чтобы рейтинг 500 не давил рейтинг 5
print('связей в графе:', len(edges))


def build_graph(e):
    """Разреженная матрица смежности; у каждой строки топ-связь нормирована в 1."""
    n = len(df)
    rows = np.concatenate([e['a'].values, e['b'].values])
    cols = np.concatenate([e['b'].values, e['a'].values])
    data = np.concatenate([e['w'].values, e['w'].values]).astype('float32')
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    rowmax = A.max(axis=1).toarray().ravel()
    rowmax[rowmax == 0] = 1
    return sparse.diags(1.0 / rowmax).dot(A).tocsr().astype('float32')


def graph_scores(A, alpha):
    """Прямые связи + alpha * связи через одного посредника, в виде плотной матрицы."""
    G = A + alpha * (A @ A) if alpha > 0 else A
    G = G.toarray().astype('float32')
    np.fill_diagonal(G, 0)
    return G


#3. итоговая матрица (на ВСЕХ связях; evaluate_hybrid строит свою)
sim = (content + LAM * graph_scores(build_graph(edges), ALPHA)).astype('float32')
print('sim:', sim.shape)

keys = [franchise_key(t) for t in df['title']]


def top_k(i, k=10, s=None, exclude=()):
    """Индексы топ-k похожих на аниме i, один тайтл на франшизу."""
    s = sim if s is None else s
    seen = {keys[i]}
    out = []
    for j in np.argsort(-s[i]):
        if j == i or j in exclude or keys[j] in seen:
            continue
        seen.add(keys[j])
        out.append(int(j))
        if len(out) == k:
            break
    return out


def find_index(title):
    low = df['title'].str.lower()
    m = df[low == title.lower()]
    if m.empty:
        m = df[low.str.contains(title.lower(), regex=False)]
    return None if m.empty else int(m.index[0])


def get_recommendations(title, n=10):
    idx = find_index(title)
    if idx is None:
        return None, []
    return df.loc[idx, 'title'], [(df.loc[j, 'title'], float(sim[idx, j])) for j in top_k(idx, n)]


if __name__ == '__main__':
    for name in ['Death Note', 'Steins;Gate']:
        found, result = get_recommendations(name)
        print(f'\nПохожие на: {found}')
        for t, s in result:
            print(f'  {t}  ({s:.2f})')