import random

import model3_3  as m

K = 10
N_SAMPLE = 1000
SEED = 42
LAMS = [0.1, 0.2, 0.4, 0.8, 1.5]   # сила влияния графа
ALPHAS = [0.5, 1.0, 2.0]               # 0 = только прямые связи, 0.5 = ещё и через посредника

# прячем 20% связей: граф строим только на оставшихся 80%
test = m.edges.sample(frac=0.2, random_state=SEED)
train = m.edges.drop(test.index)

truth = {}   # аниме множество спрятанных "правильных" соседей
for a, b in zip(test['a'], test['b']):
    truth.setdefault(a, set()).add(b)
    truth.setdefault(b, set()).add(a)

random.seed(SEED)
nodes = random.sample(sorted(truth), min(N_SAMPLE, len(truth)))
print(f'Связей: обучение {len(train)}, спрятано {len(test)}. Проверяем на {len(nodes)} аниме\n')

A_train = m.build_graph(train)

train_nb = {}
for a, b in zip(train['a'], train['b']):
    train_nb.setdefault(a, set()).add(b)
    train_nb.setdefault(b, set()).add(a)

def score(s):
    hits, recalls = [], []
    for i in nodes:
        top = set(m.top_k(i, K, s, exclude=train_nb.get(i, ())))
        overlap = top & truth[i]
        hits.append(1 if overlap else 0)
        recalls.append(len(overlap) / len(truth[i]))
    return sum(hits) / len(hits), sum(recalls) / len(recalls)


h, r = score(m.content)
print(f'только content (без графа):          hit@{K} = {h:.3f}, recall@{K} = {r:.3f}')

for alpha in ALPHAS:
    G = m.graph_scores(A_train, alpha)
    for lam in LAMS:
        h, r = score(m.content + lam * G)
        print(f'граф: LAM={lam:<4} ALPHA={alpha:<4}:        hit@{K} = {h:.3f}, recall@{K} = {r:.3f}')