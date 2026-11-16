"""Árvore, bagging e floresta, boosting (arvore.html, floresta.html, boosting.html) contra scikit-learn e R, em 1000 conjuntos.
- árvore: previsões exatas contra DecisionTreeRegressor (com ccp_alpha = α/n); quando divergem, testa random_state 0…19
  (o scikit-learn sorteia a ordem dos atributos, e empates de ganho entre X₁ e X₂ podem cair para o outro lado); e contra rpart.
- boosting: previsões depois de 1, 10 e 50 árvores contra GradientBoostingRegressor(init="zero", max_leaf_nodes=d+1), idem com as sementes; e gbm (só relato).
- floresta: só em distribuição (outro gerador): média de 1000 árvores do app contra 1000 do scikit-learn, com z-escores pela variância entre árvores.
Uso: python t1_arvores.py [arvore|boosting|floresta|todos]
"""
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from comum import RS, relatorio, roda_js
from dados import gera, grade_teste

warnings.filterwarnings('ignore')


def prepara(casos):
    for c in casos:
        pdim = len(c['X'][0])
        G = grade_teste(pdim, 23)
        c['G'] = np.column_stack([G, np.zeros(len(G))]).tolist() if pdim == 1 else G.tolist()
        c['X2'] = [x + [0.0] if pdim == 1 else x for x in c['X']]


def iguais(a, b, tol=1e-9):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) <= tol


# ------------------------------------------------------------------ árvore
JS_ARV = r"""
const t = Solver.fit(caso.X, caso.y, { maxDepth: caso.depth, minLeaf: caso.leaf, ccp: caso.alpha/caso.X.length });
const pred = q => Solver.predictTree(t, q[0], q[1]);
return { grade: caso.G.map(pred), treino: caso.X2.map(pred), folhas: t.nodes.filter(v => v.l < 0).length, rss0: t.nodes[0].imp*t.nodes[0].w };
"""


def r_rpart(casos, out):
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        L, Gl = [], []
        for k, c in enumerate(casos):
            X = np.asarray(c['X'])
            for i in range(len(X)):
                L.append(','.join(map(str, [k, c['depth'] or 30, c['leaf'], c['alpha'] / out[k]['rss0'] if out[k]['rss0'] > 0 else 0, float(X[i, 0]), float(X[i, 1]) if X.shape[1] > 1 else 'NA', float(c['y'][i])])))
            for g in c['G']:
                Gl.append(','.join(map(str, [k, float(g[0]), float(g[1]) if X.shape[1] > 1 else 'NA'])))
        a, b = os.path.join(d, 'a.csv'), os.path.join(d, 'b.csv')
        open(a, 'w').write('\n'.join(L) + '\n'); open(b, 'w').write('\n'.join(Gl) + '\n')
        scr = r"""
suppressPackageStartupMessages(library(rpart)); a <- commandArgs(TRUE); A <- read.csv(a[1], header = FALSE); G <- read.csv(a[2], header = FALSE)
for (k in unique(A$V1)) { B <- A[A$V1 == k, ]; Q <- G[G$V1 == k, ]; two <- !all(is.na(B$V6))
  dados <- if (two) data.frame(X1 = B$V5, X2 = B$V6, y = B$V7) else data.frame(X1 = B$V5, y = B$V7)
  x0 <- if (two) data.frame(X1 = Q$V2, X2 = Q$V3) else data.frame(X1 = Q$V2); L <- B$V3[1]
  m <- rpart(y ~ ., data = dados, method = "anova", control = rpart.control(minsplit = 2*L, minbucket = L, cp = -1, maxdepth = B$V2[1], maxcompete = 0, maxsurrogate = 0, xval = 0))
  if (B$V4[1] > 0) m <- prune(m, cp = B$V4[1])
  cat(k, "P", sprintf("%.17g", predict(m, x0)), "\n"); cat(k, "W", m$where, "\n") }
"""
        rf = os.path.join(d, 'r.R'); open(rf, 'w').write(scr)
        r = subprocess.run([RS, rf, a, b], capture_output=True, text=True)
    res = {}
    for ln in r.stdout.strip().splitlines():
        v = ln.split(); res[(int(v[0]), v[1])] = np.array([float(t) for t in v[2:]])
    return res


def arvore(casos):
    for i, c in enumerate(casos):
        c['depth'] = [None, 1, 2, 3, 5][i % 5]
        c['leaf'] = [1, 2, 5][(i // 5) % 3]
        c['alpha'] = [0.0, 0.05, 0.5, 2.0][(i // 15) % 4]
    out, ms = roda_js('arvore', JS_ARV, casos)
    exatos, sementes, diverg, njs, mesma_part = 0, 0, [], 0, []
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o:
            njs += 1; continue
        X, y, pdim = np.asarray(c['X']), np.asarray(c['y']), len(c['X'][0])
        G = np.asarray(c['G'])[:, :pdim]
        alvo = np.r_[o['grade'], o['treino']]
        ok = None
        for rs in range(20):
            m = DecisionTreeRegressor(criterion='squared_error', max_depth=c['depth'], min_samples_split=2, min_samples_leaf=c['leaf'], ccp_alpha=c['alpha'] / len(y), random_state=rs).fit(X, y)
            if iguais(alvo, np.r_[m.predict(G), m.predict(X)]):
                ok = rs; break
        if ok == 0: exatos += 1
        elif ok is not None: sementes += 1
        else:
            # Empate entre atributos em vários nós: a partição dos pontos de treino é a mesma, só os cortes entre eles mudam
            m = DecisionTreeRegressor(criterion='squared_error', max_depth=c['depth'], min_samples_split=2, min_samples_leaf=c['leaf'], ccp_alpha=c['alpha'] / len(y), random_state=0).fit(X, y)
            if iguais(o['treino'], m.predict(X)) and o['folhas'] == m.get_n_leaves(): mesma_part.append(i)
            else: diverg.append(i)
    sel = [i for i in range(1000) if i % 4 in (0, 1)]  # metade com uma variável, metade com duas
    R = r_rpart([casos[i] for i in sel], [out[i] for i in sel])
    JS_F = 'const t = Solver.fit(caso.X, caso.y, { maxDepth: caso.depth, minLeaf: caso.leaf, ccp: caso.alpha/caso.X.length }); return caso.X2.map(x => Solver.leafOf(t, x[0], x[1]).id);'
    folhas, _ = roda_js('arvore', JS_F, [casos[i] for i in sel], 'arvore_folhas')
    part = lambda lab: sorted(tuple(np.where(np.asarray(lab) == v)[0]) for v in np.unique(lab))
    rp_dif, rp_part, rp_rss = [], [], []
    for k, i in enumerate(sel):
        P = R.get((k, 'P'))
        if P is not None and len(P) == len(out[i]['grade']) and iguais(P, out[i]['grade'], 1e-7): continue
        W = R.get((k, 'W'))
        yv = np.asarray(casos[i]['y'])
        rss = lambda lab: sum(((yv[np.asarray(lab) == v] - yv[np.asarray(lab) == v].mean()) ** 2).sum() for v in np.unique(lab))
        if W is not None and part(W) == part(folhas[k]): rp_part.append(i)
        elif W is not None and abs(rss(W) - rss(folhas[k])) <= 1e-9 * (1 + rss(W)): rp_rss.append(i)
        else: rp_dif.append(i)
    rp_ok = len(sel) - len(rp_dif) - len(rp_part) - len(rp_rss)
    return [f'arvore.html: árvore de regressão em {len(casos)} conjuntos ({ms:.0f} ms no Node); profundidade máx. {{sem, 1, 2, 3, 5}}, mínimo por folha {{1, 2, 5}}, α {{0; 0,05; 0,5; 2}}',
            f'- erros de execução no JS: {njs}',
            f'- previsões (grade e treino) iguais às do DecisionTreeRegressor com random_state = 0: {exatos}',
            f'- iguais com outra semente (empate de ganho entre X₁ e X₂, que o scikit-learn sorteia): {sementes}',
            f'- diferentes com todas as 20 sementes, mas com as mesmas previsões nos pontos de treino e o mesmo número de folhas (empates entre X₁ e X₂ em vários nós; muda só onde o corte passa entre os pontos): {len(mesma_part)}',
            f'- diferentes de verdade: {len(diverg)}' + (f'; casos {diverg[:10]}' if diverg else ''),
            f'- rpart (method = "anova", cp = −1 e poda com cp = α/RSS da raiz), {len(sel)} conjuntos: previsões iguais na grade em {rp_ok}; diferentes na grade, mas com a mesma partição dos pontos de treino (empate de ganho entre X₁ e X₂ desempatado de outro jeito pelo arredondamento): {len(rp_part)}; outra partição com o mesmo RSS de treino (empate de ganho com x repetidos): {len(rp_rss)}; diferentes de verdade: {len(rp_dif)}' + (f' (casos {rp_dif[:12]})' if rp_dif else '')]


# ------------------------------------------------------------------ boosting
JS_BOO = r"""
const G = Solver.boost(caso.X, caso.y, { B: 50, lr: caso.lr, d: caso.d });
const est = b => caso.G.map(q => { let s = 0; for (let t = 0; t < b; t++) s += caso.lr*Solver.predictTree(G.trees[t], q[0], q[1]); return s; });
return { e1: est(1), e10: est(10), e50: est(50), train: G.train };
"""


JS_BOO_T = r"""
const G = Solver.boost(caso.X, caso.y, { B: 50, lr: caso.lr, d: caso.d });
return G.trees.map(t => t.nodes.map(v => [v.f, v.t, v.l, v.r]));
"""


def empate_anterior(c, m):
    """Primeira árvore em que app e scikit-learn separam os pontos de treino de outro jeito: as duas partições têm o mesmo RSS dos resíduos?"""
    arv = roda_js('boosting', JS_BOO_T, [c], 'boosting_arvores')[0][0]
    X, y = np.asarray(c['X']), np.asarray(c['y'])
    Xf = X.astype(np.float32)
    est = [np.zeros(len(y))] + list(m.staged_predict(X))
    part = lambda lab: sorted(tuple(np.where(lab == v)[0]) for v in np.unique(lab))
    for b, nodes in enumerate(arv):
        def folha(x):
            k = 0
            while nodes[k][2] >= 0: k = nodes[k][2] if x[nodes[k][0]] <= nodes[k][1] else nodes[k][3]
            return k
        la = np.array([folha(x) for x in Xf.astype(float)]); ls = m.estimators_[b][0].tree_.apply(Xf)
        if part(la) != part(ls):
            r = y - est[b]
            rss = lambda lab: sum(((r[lab == v] - r[lab == v].mean()) ** 2).sum() for v in np.unique(lab))
            return abs(rss(la) - rss(ls)) <= 1e-9 * (1 + rss(ls))
    return False


def boosting(casos):
    for i, c in enumerate(casos):
        c['lr'] = [0.1, 0.5, 1.0, 0.05][i % 4]
        c['d'] = [1, 2, 3, 5][(i // 4) % 4]
    out, ms = roda_js('boosting', JS_BOO, casos)
    exatos, sementes, diverg, njs, mse_ok, mesmo_treino, empate_rss = 0, 0, [], 0, 0, [], []
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o:
            njs += 1; continue
        X, y, pdim = np.asarray(c['X']), np.asarray(c['y']), len(c['X'][0])
        G = np.asarray(c['G'])[:, :pdim]
        alvo = np.r_[o['e1'], o['e10'], o['e50']]
        ok = None
        for rs in range(20):
            m = GradientBoostingRegressor(loss='squared_error', learning_rate=c['lr'], n_estimators=50, max_leaf_nodes=c['d'] + 1, max_depth=None, min_samples_leaf=1,
                                          criterion='squared_error', subsample=1.0, init='zero', random_state=rs).fit(X, y)
            st = list(m.staged_predict(G))
            if iguais(alvo, np.r_[st[0], st[9], st[49]], 1e-8):
                ok = rs
                tr = [np.mean((y - v) ** 2) for v in m.staged_predict(X)]
                mse_ok += iguais(o['train'][1:], tr, 1e-8)
                break
        if ok == 0: exatos += 1
        elif ok is not None: sementes += 1
        else:
            # Empates de ganho entre X₁ e X₂ em várias árvores: os ajustes nos pontos de treino são os mesmos (mesmo MSE a cada árvore)
            m = GradientBoostingRegressor(loss='squared_error', learning_rate=c['lr'], n_estimators=50, max_leaf_nodes=c['d'] + 1, max_depth=None, min_samples_leaf=1,
                                          criterion='squared_error', subsample=1.0, init='zero', random_state=0).fit(X, y)
            tr = [np.mean((y - v) ** 2) for v in m.staged_predict(X)]
            if iguais(o['train'][1:], tr, 1e-10): mesmo_treino.append(i)
            elif empate_anterior(c, m): empate_rss.append(i)
            else: diverg.append(i)
    return [f'boosting.html: Algoritmo 8.2 em {len(casos)} conjuntos ({ms:.0f} ms no Node); B = 50, λ ∈ {{0,05; 0,1; 0,5; 1}}, d ∈ {{1, 2, 3, 5}}',
            f'- erros de execução no JS: {njs}',
            f'- previsões depois de 1, 10 e 50 árvores iguais às do GradientBoostingRegressor(init="zero") com random_state = 0: {exatos}',
            f'- iguais com outra semente (empate de ganho entre atributos): {sementes}',
            f'- MSE de treino a cada árvore igual ao do scikit-learn (nos casos iguais): {mse_ok} de {exatos + sementes}',
            f'- diferentes com todas as 20 sementes, mas com o mesmo MSE de treino a cada árvore (empates entre X₁ e X₂ que separam os mesmos pontos, em várias árvores): {len(mesmo_treino)}',
            f'- diferentes porque, numa árvore, app e scikit-learn escolheram partições diferentes com o mesmo RSS dos resíduos (empate com x repetidos), o que muda as árvores seguintes: {len(empate_rss)}',
            f'- diferentes de verdade: {len(diverg)}' + (f'; casos {diverg[:10]}' if diverg else '')]


# ------------------------------------------------------------------ floresta
JS_FLO = r"""
let soma = null, soma2 = null, k = 0, oob = 0;
for (const seed of caso.seeds) {
  const F = Solver.fit(caso.X, caso.y, { B: caso.B, seed, base: caso.base, maxDepth: null, minLeaf: 1, m: caso.m });
  for (let b = 0; b < F.B; b++) {
    const v = caso.G.map(q => Solver.predictOne(F, b, q[0], q[1]));
    if (!soma) { soma = v.map(() => 0); soma2 = v.map(() => 0); }
    v.forEach((u, j) => { soma[j] += u; soma2[j] += u*u; }); k++;
  }
  oob += Solver.oob(F, caso.X2, caso.y).mse/caso.seeds.length;
}
return { media: soma.map(s => s/k), var: soma2.map((s, j) => s/k - (soma[j]/k)**2), k, oob };
"""


def floresta(casos):
    sel = [i for i in range(1000) if i % 5 == 0 and len(casos[i]['y']) >= 5][:200]
    cs = []
    for k, i in enumerate(sel):
        c = dict(casos[i]); pdim = len(c['X'][0])
        c['base'] = 'linear' if k % 4 == 3 else 'tree'
        c['m'] = 1 if pdim == 2 and k % 2 == 0 else pdim
        c['B'] = 200; c['seeds'] = [11, 22, 33, 44, 55]
        c['G'] = c['G'][::3]
        cs.append(c)
    out, ms = roda_js('floresta', JS_FLO, cs)
    zs, oobd, njs, sem_oob = [], [], 0, [0]
    for c, o in zip(cs, out):
        if 'erro' in o:
            njs += 1; continue
        X, y, pdim = np.asarray(c['X']), np.asarray(c['y']), len(c['X'][0])
        G = np.asarray(c['G'])[:, :pdim]
        P, oo = [], []
        for rs in range(5):
            if c['base'] == 'linear':
                try:
                    m = BaggingRegressor(estimator=LinearRegression(), n_estimators=200, bootstrap=True, oob_score=True, random_state=rs).fit(X, y)
                except ValueError:  # alguma amostra bootstrap sem pontos de fora: o scikit-learn não calcula o OOB
                    m = BaggingRegressor(estimator=LinearRegression(), n_estimators=200, bootstrap=True, random_state=rs).fit(X, y)
                    m.oob_prediction_ = np.full(len(y), np.nan); sem_oob[0] += 1
            else:
                m = RandomForestRegressor(n_estimators=200, max_features=c['m'], max_depth=None, min_samples_leaf=1, bootstrap=True, oob_score=True, random_state=rs).fit(X, y)
            P += [e.predict(G[:, ] if c['base'] == 'tree' else G) for e in m.estimators_]
            cob = np.isfinite(m.oob_prediction_)
            if np.all(np.isfinite(m.oob_prediction_)): oo.append(np.mean((y - m.oob_prediction_) ** 2))
        P = np.array(P); ms_, vs = P.mean(0), P.var(0)
        se = np.sqrt((vs + np.array(o['var'])) / len(P) + 1e-18)
        z = np.abs(np.array(o['media']) - ms_) / se
        zs.append(float(np.max(np.where(se > 1e-9, z, 0))))
        if oo: oobd.append((o['oob'], float(np.mean(oo))))
    zs = np.array(zs); nz = int(np.sum(zs > 5))
    ob = np.array(oobd); rel = np.abs(ob[:, 0] - ob[:, 1]) / (1e-9 + ob[:, 1])
    return [f'floresta.html: {len(cs)} conjuntos, em distribuição ({ms:.0f} ms no Node): 1000 árvores (5 sementes × B = 200) do app contra 1000 do scikit-learn; base árvore (m = 1 ou p) ou regressão linear',
            f'- erros de execução no JS: {njs}',
            f'- maior |z| da diferença das médias por ponto da grade (z pela variância entre árvores): mediana {np.median(zs):.2f}, máximo {zs.max():.2f}; conjuntos com |z| > 5: {nz}',
            f'- MSE OOB (média de 5 sementes) do app contra o do scikit-learn: diferença relativa mediana {np.median(rel):.3f}, máxima {rel.max():.3f} (é aleatório; só a ordem de grandeza tem de bater). BaggingRegressor com oob_score=True falhou (amostra bootstrap sem pontos de fora) em {sem_oob[0]} ajustes']


def main(qual):
    casos = gera()
    prepara(casos)
    linhas = []
    if qual in ('arvore', 'todos'): linhas += arvore(casos)
    if qual in ('boosting', 'todos'): linhas += boosting(casos)
    if qual in ('floresta', 'todos'): linhas += floresta(casos)
    relatorio(f't1_arvores_{qual}', linhas)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'todos')
