"""
Запускается один раз (и заново, когда меняю модель или данные).
Считает итоговую выдачу для каждого аниме и сохраняет:

  data/recs_index.npz   топ-K соседей (индексы и оценки) для каждого аниме
  data/meta.csv         название, жанры, год, оценка, обложка, ссылка

Приложение читает только эти два маленьких файла.
"""

import numpy as np

import model3_3 as m

K = 30
n = len(m.df)

neighbors = np.full((n, K), -1, dtype='int32')   # -1 = пусто (на случай, если соседей меньше K)
scores = np.zeros((n, K), dtype='float32')

for i in range(n):
    top = m.top_k(i, K)
    neighbors[i, :len(top)] = top
    scores[i, :len(top)] = m.sim[i, top]

np.savez_compressed('data/recs_index.npz', neighbors=neighbors, scores=scores)

cols = ['id', 'title', 'title_english', 'genres', 'year', 'score', 'popularity', 'cover', 'url']
m.df[[c for c in cols if c in m.df.columns]].to_csv('data/meta.csv', index=False)

print(f'Готово: {n} аниме, по {K} соседей. Файлы: data/recs_index.npz, data/meta.csv')