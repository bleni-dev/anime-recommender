"""
Лёгкий модуль для приложения: не считает ничего, только читает готовые файлы.
Нужны лишь pandas и numpy.
"""

import numpy as np
import pandas as pd

meta = pd.read_csv('data/meta.csv')
_index = np.load('data/recs_index.npz')
neighbors = _index['neighbors']
scores = _index['scores']

df = meta  # то же имя, что в старой модели, чтобы app.py (df['title']) продолжал работать
_lower = meta['title'].str.lower()


def find_index(title):
    m = meta[_lower == title.lower()]
    if m.empty:
        m = meta[_lower.str.contains(title.lower(), regex=False)]
    return None if m.empty else int(m.index[0])


def get_recommendations(title, n=10):
    """Тот же формат ответа, что и раньше: (найденное название, [(название, оценка), ...])."""
    i = find_index(title)
    if i is None:
        return None, []
    n = min(n, neighbors.shape[1])
    result = [
        (meta.loc[j, 'title'], float(s))
        for j, s in zip(neighbors[i, :n], scores[i, :n])
        if j >= 0
    ]
    return meta.loc[i, 'title'], result