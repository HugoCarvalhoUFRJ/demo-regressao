"""Mínimos quadrados (linear.html): o solver do app contra numpy/scikit-learn, statsmodels e o lm do R, em 1000 conjuntos.
Uso: python t1_linear.py
"""
import os
import subprocess
import tempfile
import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from comum import RS, relatorio, roda_js
from dados import gera

warnings.filterwarnings('ignore')
GRAUS = [1, 2, 3, 5, 10]

JS = r"""
const Z = Solver.design(caso.X, caso.d, caso.inter), r = Solver.ols(Z, caso.y);
const A = a => a ? Array.from(a) : null;
return { beta: A(r.beta), fitted: A(r.fitted), rss: r.rss, se: A(r.se), t: A(r.t), p: A(r.p), F: r.F ?? null, pF: r.pF ?? null,
         h: A(r.h), stud: A(r.stud), perfeito: !!r.perfect, minnorm: !!r.minnorm, rank: r.rank, df: r.df, rse: r.rse ?? null, r2: r.r2 };
"""


def desenho(X, d, inter):
    X = np.asarray(X)
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0] ** k for k in range(d + 1)])
    cols = [np.ones(len(X)), X[:, 0], X[:, 1]] + ([X[:, 0] * X[:, 1]] if inter else [])
    return np.column_stack(cols)


def r_lm(casos):
    """Coeficientes e erros-padrão do lm, para os casos de posto completo."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        linhas = []
        for k, c in enumerate(casos):
            Z = desenho(c['X'], c['d'], c['inter'])
            for i in range(len(Z)):
                linhas.append(','.join([str(k)] + [repr(float(v)) for v in Z[i, 1:]] + [repr(float(c['y'][i]))]))
        arq = os.path.join(d, 'z.csv')
        with open(arq, 'w') as f:
            f.write('\n'.join(linhas))
        scr = r"""
args <- commandArgs(TRUE); L <- readLines(args[1]); parts <- strsplit(L, ","); id <- as.integer(sapply(parts, `[`, 1))
for (k in unique(id)) {
  M <- do.call(rbind, lapply(parts[id == k], function(v) as.numeric(v[-1]))); q <- ncol(M) - 1
  Z <- M[, 1:q, drop = FALSE]; y <- M[, q + 1]
  m <- lm(y ~ Z); s <- summary(m)$coefficients
  cat(k, ifelse(any(is.na(coef(m))), "NA", paste(c(s[, 1], s[, 2]), collapse = " ")), "\n")
}
"""
        rf = os.path.join(d, 'r.R')
        with open(rf, 'w') as f:
            f.write(scr)
        out = subprocess.run([RS, rf, arq], capture_output=True, text=True).stdout
    res = {}
    for ln in out.strip().splitlines():
        parts = ln.split()
        res[int(parts[0])] = None if parts[1] == 'NA' else np.array([float(v) for v in parts[1:]])
    return res


def stud_mp(Z, y, k):
    import mpmath as mp
    mp.mp.dps = 60
    Zm, ym = mp.matrix(Z.tolist()), mp.matrix(y.tolist())
    H = Zm * mp.inverse(Zm.T * Zm) * Zm.T
    e = ym - H * ym
    rss, df, h = sum(e[j] ** 2 for j in range(len(y))), len(y) - Z.shape[1], H[k, k]
    return float(e[k] / mp.sqrt((rss - e[k] ** 2 / (1 - h)) / (df - 1) * (1 - h)))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / (1 + np.abs(b)))) if a.size else 0.0


def main():
    casos = gera()
    for i, c in enumerate(casos):
        p = len(c['X'][0])
        c['d'] = GRAUS[i % 5] if p == 1 else 1
        c['inter'] = p == 2 and (i // 2) % 2 == 1
    out, ms = roda_js('linear', JS, casos)
    falhas = {k: [] for k in ['ajustados', 'coef', 'minnorm', 'posto', 'se', 'p', 'F', 'h', 'stud', 'lm']}
    maior = {k: 0.0 for k in falhas}
    erros_js = [i for i, o in enumerate(out) if 'erro' in o]
    sel_r = []
    arbitro = {'app': 0, 'nan': 0}
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o:
            continue
        Z = desenho(c['X'], c['d'], c['inter']); y = np.array(c['y'])
        lr = LinearRegression().fit(Z[:, 1:], y)
        ref_b = np.r_[lr.intercept_, lr.coef_]
        fit_ref = Z @ ref_b
        escala = 1 + np.max(np.abs(y))
        e = float(np.max(np.abs(np.array(o['fitted']) - fit_ref))) / escala
        maior['ajustados'] = max(maior['ajustados'], e)
        if e > 1e-7: falhas['ajustados'].append((i, e))
        posto = np.linalg.matrix_rank(Z)
        if (posto < Z.shape[1]) != o['minnorm']:
            falhas['posto'].append((i, posto, o['rank'], c['d']))
            continue
        if o['minnorm']:
            e = rel(o['beta'], ref_b); maior['minnorm'] = max(maior['minnorm'], e)
            if e > 1e-6: falhas['minnorm'].append((i, e))
            continue
        e = rel(o['beta'], ref_b); maior['coef'] = max(maior['coef'], e)
        if e > 1e-6: falhas['coef'].append((i, e, c['d']))
        if o['df'] > 0 and not o['perfeito']:
            ols = sm.OLS(y, Z).fit()
            for nome, ref, val in [('se', ols.bse, o['se']), ('p', ols.pvalues, o['p'])]:
                if val is None: continue
                e = rel(val, ref); maior[nome] = max(maior[nome], e)
                if e > 1e-6: falhas[nome].append((i, e))
            if o['F'] is not None and np.isfinite(ols.fvalue):
                e = rel([o['F'], o['pF']], [ols.fvalue, ols.f_pvalue]); maior['F'] = max(maior['F'], e)
                if e > 1e-6: falhas['F'].append((i, e))
            inf = ols.get_influence()
            e = rel(o['h'], inf.hat_matrix_diag); maior['h'] = max(maior['h'], e)
            if e > 1e-8: falhas['h'].append((i, e))
            if o['stud'] is not None and o['df'] > 1:
                ref = inf.resid_studentized_external
                ok = np.isfinite(ref) & np.isfinite(np.array(o['stud'], float))
                js = np.array(o['stud'], float)
                if ok.any():
                    e = rel(js[ok], ref[ok])
                    if e > 1e-6:
                        # Divergência: o árbitro é a conta em alta precisão (mpmath, 60 dígitos)
                        k = int(np.argmax(np.where(ok, np.abs(js - ref) / (1 + np.abs(ref)), -1)))
                        tm = stud_mp(Z, y, k)
                        if abs(js[k] - tm) <= abs(ref[k] - tm): arbitro['app'] += 1
                        else: falhas['stud'].append((i, e)); maior['stud'] = max(maior['stud'], e)
                    else: maior['stud'] = max(maior['stud'], e)
                arbitro['nan'] += int(np.sum(~np.isfinite(js) & np.isfinite(ref)))
            if c['d'] <= 5:
                sel_r.append(i)
    r = r_lm([casos[i] for i in sel_r])
    n_na = 0
    for k, i in enumerate(sel_r):
        o, v = out[i], r.get(k)
        if v is None:
            n_na += 1
            continue
        q = len(o['beta'])
        e = rel(np.r_[o['beta'], o['se']], v[:2 * q]) if len(v) == 2 * q else 1.0
        maior['lm'] = max(maior['lm'], e)
        if e > 1e-6: falhas['lm'].append((i, e))
    linhas = [f'linear.html: solver de mínimos quadrados em {len(casos)} conjuntos ({ms:.0f} ms no Node; graus {GRAUS} com uma variável, com e sem interação com duas)',
              f'erros de execução no JS: {len(erros_js)}']
    nomes = {'ajustados': 'valores ajustados contra LinearRegression', 'posto': 'decisão de posto incompleto contra numpy.matrix_rank',
             'coef': 'coeficientes (posto completo) contra LinearRegression', 'minnorm': 'coeficientes de menor norma (posto incompleto) contra LinearRegression',
             'se': 'erros-padrão contra statsmodels', 'p': 'p-valores t contra statsmodels', 'F': 'F e p-valor contra statsmodels',
             'h': 'alavancagem contra statsmodels', 'stud': 'resíduo studentizado externo contra statsmodels', 'lm': f'coeficientes e erros-padrão contra o lm do R ({len(sel_r)} casos com grau ≤ 5)'}
    for k, nm in nomes.items():
        linhas.append(f'- {nm}: {len(falhas[k])} divergências; maior diferença relativa {maior[k]:.2e}' + (f'; exemplos {falhas[k][:4]}' if falhas[k] else ''))
    linhas.append(f"- resíduo studentizado: em {arbitro['app']} casos o app e o statsmodels diferem e o app está mais perto da conta em alta precisão (mpmath); {arbitro['nan']} valores que o app deixa sem número (hᵢ > 1 − 10⁻¹⁰, conta indefinida) e o statsmodels calcula")
    linhas.append(f'- lm do R com coeficiente NA (o lm considerou a matriz singular, tolerância 1e-7): {n_na} casos')
    relatorio('t1_linear', linhas)


if __name__ == '__main__':
    main()
