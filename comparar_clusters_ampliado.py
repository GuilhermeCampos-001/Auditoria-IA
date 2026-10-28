"""Compara uma faixa ampliada de quantidades de clusters no Recomovie AI.

Execute na pasta do projeto:
    python comparar_clusters_ampliado.py

Testa k = 60, 75, 80, 90, 100, 125, 150 e 200.
Gera comparacao_clusters_ampliada.csv e dois gráficos PNG.
O script NÃO substitui nem salva o modelo definitivo.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from recomendador import RecomendadorFilmes

PASTA = Path(__file__).resolve().parent
VALORES_K = [60, 75, 80, 90, 100, 125, 150, 200]
QUANTIDADE_FILMES_TESTE = 30
RECOMENDACOES_POR_FILME = 10
MINIMO_AVALIACOES = 50


def main():
    print("Carregando dados e preparando características...")
    base = RecomendadorFilmes()
    X = base.X

    generos_reais = [
        g for g in base.generos
        if g not in {"(no genres listed)", "no_genres_list"}
    ]
    fontes = base.dados[base.dados[generos_reais].sum(axis=1) > 0]
    fontes = fontes.sample(
        n=min(QUANTIDADE_FILMES_TESTE, len(fontes)), random_state=42
    )
    resultados = []

    for k in VALORES_K:
        print(f"\nTestando {k} clusters...")
        modelo = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=2048, n_init=3
        )
        grupos = modelo.fit_predict(X)

        silhueta = silhouette_score(
            X, grupos, sample_size=min(10000, len(X)), random_state=42
        )
        davies = davies_bouldin_score(X, grupos)
        calinski = calinski_harabasz_score(X, grupos)

        tamanhos = pd.Series(grupos).value_counts().reindex(range(k), fill_value=0)
        tamanhos_validos = tamanhos[tamanhos > 0]

        indicadores_genero = []
        total_recomendacoes = 0
        filmes_sem_resultado = 0
        movie_ids = base.dados["movieId"].to_numpy()
        quantidade_avaliadores = base.dados["quant_avaliadores"].to_numpy()

        for indice, fonte in fontes.iterrows():
            cluster_fonte = grupos[indice]
            mascara = (grupos == cluster_fonte) & (movie_ids != fonte["movieId"])
            mascara &= quantidade_avaliadores >= MINIMO_AVALIACOES
            candidatos = np.flatnonzero(mascara)

            if len(candidatos) == 0:
                filmes_sem_resultado += 1
                continue

            distancias = pairwise_distances(
                X[indice].reshape(1, -1), X[candidatos], metric="cosine"
            )[0]
            ordem = np.argsort(distancias)[:RECOMENDACOES_POR_FILME]
            recomendados = candidatos[ordem]
            generos_fonte = (
                base.dados.iloc[indice][generos_reais].to_numpy(dtype=float) > 0
            )

            for indice_recomendado in recomendados:
                generos_recomendado = (
                    base.dados.iloc[indice_recomendado][generos_reais]
                    .to_numpy(dtype=float) > 0
                )
                indicadores_genero.append(
                    bool(np.logical_and(generos_fonte, generos_recomendado).any())
                )
                total_recomendacoes += 1

        qualidade = (
            100 * np.mean(indicadores_genero) if indicadores_genero else 0.0
        )
        resultados.append({
            "clusters_k": k,
            "silhouette_score": silhueta,
            "davies_bouldin": davies,
            "calinski_harabasz": calinski,
            "tamanho_minimo": int(tamanhos_validos.min()),
            "tamanho_mediano": float(tamanhos_validos.median()),
            "tamanho_maximo": int(tamanhos_validos.max()),
            "clusters_vazios": int((tamanhos == 0).sum()),
            "recomendacoes_avaliadas": total_recomendacoes,
            "recomendacoes_com_genero_compartilhado_pct": qualidade,
            "filmes_teste_sem_recomendacoes": filmes_sem_resultado,
        })

        print(
            f"Silhouette={silhueta:.4f} | Davies-Bouldin={davies:.4f} | "
            f"Calinski-Harabasz={calinski:.2f} | "
            f"Gênero compartilhado={qualidade:.1f}%"
        )
        print(
            f"Tamanhos: mínimo={tamanhos_validos.min()}, "
            f"mediana={tamanhos_validos.median():.0f}, "
            f"máximo={tamanhos_validos.max()}, "
            f"clusters vazios={(tamanhos == 0).sum()}"
        )

    df = pd.DataFrame(resultados)
    arquivo_csv = PASTA / "comparacao_clusters_ampliada.csv"
    df.to_csv(arquivo_csv, index=False, encoding="utf-8-sig")

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    eixos[0].plot(df["clusters_k"], df["silhouette_score"], marker="o")
    eixos[0].set_title("Silhouette Score (maior é melhor)")
    eixos[0].set_xlabel("Número de clusters (k)")
    eixos[0].set_ylabel("Silhouette Score")
    eixos[0].grid(True, alpha=0.3)

    eixos[1].plot(df["clusters_k"], df["davies_bouldin"], marker="o")
    eixos[1].set_title("Davies-Bouldin (menor é melhor)")
    eixos[1].set_xlabel("Número de clusters (k)")
    eixos[1].set_ylabel("Índice Davies-Bouldin")
    eixos[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / "metricas_clusters_ampliada.png", dpi=150)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(9, 4.5))
    eixo.plot(
        df["clusters_k"],
        df["recomendacoes_com_genero_compartilhado_pct"],
        marker="o",
    )
    eixo.set_title("Recomendações com pelo menos um gênero em comum")
    eixo.set_xlabel("Número de clusters (k)")
    eixo.set_ylabel("Percentual (%)")
    eixo.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / "qualidade_recomendacoes_ampliada.png", dpi=150)
    plt.close(fig)

    print("\n=== COMPARAÇÃO FINAL ===")
    print(df.round(3).to_string(index=False))
    print("\nArquivos gerados:")
    print(f"- {arquivo_csv.name}")
    print("- metricas_clusters_ampliada.png")
    print("- qualidade_recomendacoes_ampliada.png")
    print(
        "\nObservação: gênero compartilhado é uma medida simples; não comprova "
        "que as recomendações sejam realmente relevantes."
    )


if __name__ == "__main__":
    main()
