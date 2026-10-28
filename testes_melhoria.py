"""Testes de regressao da melhoria de validacao e tratamento de generos ausentes."""
from pathlib import Path
import pandas as pd
from recomendador import RecomendadorFilmes

PASTA = Path(__file__).resolve().parent
rec = RecomendadorFilmes()
linhas = []

def testar(nome, movie_id, quantidade=10, minimo=50, esperado=None):
    try:
        resultado = rec.recomendar(movie_id, quantidade, minimo)
        passou = esperado(resultado) if esperado else True
        detalhe = f"{len(resultado)} recomendacoes"
        status = "PASSOU" if passou else "FALHOU"
    except Exception as exc:
        status = "FALHOU"
        detalhe = f"Excecao: {type(exc).__name__}: {exc}"
    linhas.append((nome, status, detalhe))
    print(f"[{status}] {nome}: {detalhe}")

# ID válido, entradas inválidas, parâmetros inválidos e filme sem gênero.
testar("Filme válido Toy Story (ID 1)", 1, esperado=lambda r: len(r) > 0)
testar("ID inexistente", 999999999, esperado=lambda r: r.empty)
testar("ID string", "abc", esperado=lambda r: r.empty)
testar("ID None", None, esperado=lambda r: r.empty)
testar("ID float não inteiro", 3.14, esperado=lambda r: r.empty)
testar("ID negativo", -5, esperado=lambda r: r.empty)
testar("Quantidade zero", 1, quantidade=0, esperado=lambda r: r.empty)
testar("Mínimo de avaliações negativo", 1, minimo=-1, esperado=lambda r: r.empty)

sem_genero = rec.dados[rec.dados[rec.generos].sum(axis=1) == 0]
if not sem_genero.empty:
    linha = sem_genero.iloc[0]
    testar(f"Filme sem gêneros ({linha['title']})", int(linha['movieId']), esperado=lambda r: r.empty)
else:
    linhas.append(("Filme sem gêneros", "N/A", "Não há filmes sem gêneros na base"))

# Verificação simples de coerência por gênero em uma amostra reprodutível.
fontes = rec.dados[rec.dados[rec.generos].sum(axis=1) > 0].sample(n=30, random_state=42)
avaliadas = 0
compartilhadas = 0
sem_resultados = 0
for _, fonte in fontes.iterrows():
    resultado = rec.recomendar(int(fonte['movieId']), quantidade=10)
    if resultado.empty:
        sem_resultados += 1
        continue
    generos_fonte = set(g for g in rec.generos if fonte[g] > 0)
    for _, recomendado in resultado.iterrows():
        generos_rec = set(g for g in rec.generos if rec.dados.loc[rec.dados['movieId'] == recomendado['movieId'], g].iloc[0] > 0)
        avaliadas += 1
        if generos_fonte & generos_rec:
            compartilhadas += 1
pct = (100 * compartilhadas / avaliadas) if avaliadas else 0
linhas.append(("Coerência de gênero (amostra de 30 filmes)", "INFO", f"{compartilhadas}/{avaliadas} ({pct:.1f}%) com gênero compartilhado; {sem_resultados} filmes sem resultado"))
print(f"[INFO] Coerência de gênero: {compartilhadas}/{avaliadas} ({pct:.1f}%); sem resultados: {sem_resultados}")

saida = PASTA / "AUDITORIA" / "resultados_testes_melhoria.txt"
saida.write_text("TESTES DE REGRESSÃO — RECOMOVIE AI\n\n" + "\n".join(f"[{s}] {n}: {d}" for n,s,d in linhas) + "\n", encoding="utf-8")
if any(s == "FALHOU" for _,s,_ in linhas):
    raise SystemExit("Um ou mais testes falharam")
print(f"Resultados salvos em: {saida}")
