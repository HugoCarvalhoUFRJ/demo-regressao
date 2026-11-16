"""Confere se a casca dos apps de regressão é idêntica à do knn.html (referência). Com 'aplicar', copia a casca do knn para os outros.
Uso: python t_casca.py [aplicar]
Regiões da casca: script do tema, CSS, barra e linha dos dados, narração e gráfico, primeira seção do painel e seção de código, e o script da casca.
"""
import sys

from comum import PROJ

REF = 'knn'
APPS = [a for a in ['linear', 'penalizada', 'knn', 'arvore', 'floresta', 'boosting'] if (PROJ / f'{a}.html').exists()]
REGIOES = {
    'tema': ('<script>\n// O tema', '</script>'),
    'css': ('<style>\n/* Layout', '</style>'),
    'barra': ('    <div class="bar">', '\n    <div class="params">'),
    'grafico': ('    <div id="say"', '  </main>'),
    'painel': ('  <aside>', '    <section>\n      <p class="lab" id="v-rulelab">'),
    'codigo': ('    <section>\n      <p class="lab">Mesmo modelo em código</p>', '  </aside>'),
    'script': ('(() => {\n  const X0 = -1.73', None),
}


def trecho(s, ini, fim):
    a = s.index(ini)
    b = len(s) if fim is None else s.index(fim, a)
    return a, b


def confere():
    ref = (PROJ / f'{REF}.html').read_text(encoding='utf-8')
    ok = True
    for app in APPS:
        s = (PROJ / f'{app}.html').read_text(encoding='utf-8')
        for nome, (ini, fim) in REGIOES.items():
            a, b = trecho(ref, ini, fim)
            c, d = trecho(s, ini, fim)
            if ref[a:b] != s[c:d]:
                ok = False
                print(f'{app}: região {nome} difere da do {REF}.html')
        if '/*SOLVER-START*/' not in s or 'const App = {' not in s:
            ok = False
            print(f'{app}: falta o bloco do Solver ou o App')
    nuc = [a for a in ['arvore', 'floresta', 'boosting'] if (PROJ / f'{a}.html').exists()]
    trechos = set()
    for a in nuc:
        s = (PROJ / f'{a}.html').read_text(encoding='utf-8')
        i, j = s.index('// NÚCLEO-ÁRVORE-INÍCIO'), s.index('// NÚCLEO-ÁRVORE-FIM')
        trechos.add(s[i:j])
    if len(trechos) > 1:
        ok = False
        print('o núcleo da árvore difere entre', ', '.join(nuc))
    else:
        print('núcleo da árvore idêntico em', ', '.join(nuc))
    print('casca idêntica em', ', '.join(APPS) if ok else '(com diferenças)')
    return ok


def aplica():
    ref = (PROJ / f'{REF}.html').read_text(encoding='utf-8')
    for app in APPS:
        if app == REF:
            continue
        s = (PROJ / f'{app}.html').read_text(encoding='utf-8')
        for nome, (ini, fim) in REGIOES.items():
            a, b = trecho(ref, ini, fim)
            c, d = trecho(s, ini, fim)
            s = s[:c] + ref[a:b] + s[d:]
        (PROJ / f'{app}.html').write_text(s, encoding='utf-8')
        print('aplicado em', app)


if __name__ == '__main__':
    if sys.argv[1:] == ['aplicar']:
        aplica()
    sys.exit(0 if confere() else 1)
