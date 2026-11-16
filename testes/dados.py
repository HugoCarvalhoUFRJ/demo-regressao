"""Gera exatamente 1000 conjuntos de dados de regressão com variedade planejada, metade com uma variável e metade com duas.
Tipos: 250 lineares, 250 não lineares, 200 aleatórios, 200 em grade (x repetidos e empates em y), 100 casos extremos feitos à mão.
Cada caso: {'tipo', 'X': n × p, 'y': n}. Os x ficam dentro do gráfico dos apps; os y, entre −1,5 e 3,5.
"""
import numpy as np

X0, X1, Y0, Y1 = -1.70, 3.49, -1.74, 3.72
PLANO = [('linear', 250), ('naolinear', 250), ('aleatorio', 200), ('grade', 200), ('extremo', 100)]


def _x(rng, n, p):
    return np.column_stack([rng.uniform(X0, X1, n), rng.uniform(Y0, Y1, n)])[:, :p]


def _clip(y):
    return np.clip(y, -1.5, 3.5)


def extremo(i, rng):
    """Casos degenerados: n mínimo, x constante, colineares, duplicatas, y constante, outlier, alavancagem."""
    p = 1 + i % 2
    k = (i // 2) % 10
    if k == 0:   # dois pontos
        X = _x(rng, 2, p); y = rng.uniform(-1, 3, 2)
    elif k == 1:  # três pontos
        X = _x(rng, 3, p); y = rng.uniform(-1, 3, 3)
    elif k == 2:  # x constante (na primeira coluna)
        X = _x(rng, 12, p); X[:, 0] = 0.5; y = rng.uniform(-1, 3, 12)
    elif k == 3:  # pontos duplicados
        base = _x(rng, 5, p); X = np.repeat(base, 3, axis=0); y = _clip(np.repeat(rng.uniform(-1, 3, 5), 3) + rng.normal(0, .2, 15))
    elif k == 4:  # y constante
        X = _x(rng, 15, p); y = np.full(15, 1.25)
    elif k == 5:  # colineares: X₂ = X₁ (com duas variáveis) ou poucos x distintos
        X = _x(rng, 14, p)
        if p == 2: X[:, 1] = X[:, 0]
        else: X[:, 0] = rng.choice([0.0, 1.0, 2.0], 14)
        y = _clip(1 + 0.5 * X[:, 0] + rng.normal(0, .3, 14))
    elif k == 6:  # outlier em y
        X = _x(rng, 20, p); y = _clip(0.5 + 0.4 * X[:, 0] + rng.normal(0, .2, 20)); y[3] = 3.4
    elif k == 7:  # ponto de alta alavancagem
        X = np.vstack([rng.uniform(-1, 1, (19, p)), np.full((1, p), 3.4)]); y = _clip(0.5 + 0.4 * X[:, 0] + rng.normal(0, .2, 20)); y[-1] = -1.2
    elif k == 8:  # muitos pontos
        X = _x(rng, 60, p); y = _clip(1 + np.sin(2 * X[:, 0]) + rng.normal(0, .3, 60))
    else:        # quatro pontos nos cantos de um quadrado, y em xadrez (empates de ganho nas árvores)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)[:, :p]; y = np.array([0, 1, 1, 0], float)
    return X, y


def gera(semente=2026):
    rng = np.random.default_rng(semente)
    casos = []
    for tipo, quantos in PLANO:
        for i in range(quantos):
            p = 1 + i % 2
            n = int(rng.integers(5, 61))
            if tipo == 'linear':
                X = _x(rng, n, p); b = rng.normal(0, .6, p); y = _clip(1 + X @ b + rng.normal(0, rng.uniform(0, .6), n))
            elif tipo == 'naolinear':
                X = _x(rng, n, p); f = [np.sin(2 * X[:, 0]), (X[:, 0] - 1) ** 2 / 3, (X[:, 0] > 1) * 1.5][i % 3]
                if p == 2: f = f + 0.5 * np.cos(1.5 * X[:, 1])
                y = _clip(0.5 + f + rng.normal(0, .3, n))
            elif tipo == 'aleatorio':
                X = _x(rng, n, p); y = rng.uniform(-1.5, 3.5, n)
            elif tipo == 'grade':
                g = np.arange(-1.5, 3.5, 0.5)
                X = rng.choice(g, (n, p)); y = np.round(rng.uniform(-1, 3, n) * 4) / 4
            else:
                X, y = extremo(i, rng)
            casos.append({'tipo': tipo, 'X': np.asarray(X, float).tolist(), 'y': np.asarray(y, float).tolist()})
    assert len(casos) == 1000
    return casos


def grade_teste(p, k=41):
    """Pontos onde comparar as previsões: uma grade dentro do gráfico."""
    if p == 1:
        return np.linspace(X0, X1, 4 * k)[:, None]
    a, b = np.meshgrid(np.linspace(X0, X1, k), np.linspace(Y0, Y1, k))
    return np.column_stack([a.ravel(), b.ravel()])
