"""Validação da qualidade da clusterização usada pelo RecomendadorFilmes.

Executa duas análises principais:
1. Métricas internas de clusterização (Silhouette Score, Davies-Bouldin Index e Calinski-Harabasz).
2. Estudo de sensibilidade para diferentes valores de k com gráfico de curva de Silhouette.
"""

# Importa matplotlib para geração de gráficos estatísticos
import matplotlib.pyplot as plt

# Importa pandas para manipulação de tabelas de dados
import pandas as pd

# Importa as métricas de validação de clusterização do scikit-learn
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Importa o algoritmo MiniBatchKMeans para os testes de sensibilidade de k
from sklearn.cluster import MiniBatchKMeans

# Importa a classe do recomendador de filmes
from recomendador import RecomendadorFilmes


# ==============================================================================
# 1. Avaliação de Métricas de Qualidade Interna (k = 25)
# ==============================================================================

# Instancia o recomendador (carrega os dados e treina o modelo com k=25)
recomendador = RecomendadorFilmes()

# Obtém a matriz de características numéricas normalizadas (self.X)
X = recomendador.X

# Obtém a série contendo os identificadores de cluster atribuídos a cada filme
clusters = recomendador.dados["cluster"]

# Calcula o Silhouette Score com amostragem de 10.000 pontos para agilidade e reprodutibilidade
silhueta = silhouette_score(X, clusters, sample_size=10000, random_state=42)

# Calcula o índice Davies-Bouldin (quanto menor, mais separados e coesos os clusters)
davies = davies_bouldin_score(X, clusters)

# Calcula o score Calinski-Harabasz (razão de dispersão entre e dentro dos clusters; quanto maior, melhor)
calinski = calinski_harabasz_score(X, clusters)

# Exibe os resultados no console
print("========================================")
print("       VALIDAÇÃO DO MODELO (k=25)       ")
print("========================================")
print(f"Silhouette Score    : {silhueta:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print(f"Calinski-Harabasz   : {calinski:.2f}")
print("========================================")


# ==============================================================================
# 2. Análise de Sensibilidade para Variação de k
# ==============================================================================

# Define a lista de valores de k a serem comparados
valores_k = [5, 10, 15, 20, 25, 30]

# Lista para armazenar o Silhouette Score correspondente a cada k
resultados = []

print("\nIniciando teste de sensibilidade para diferentes valores de k...")

# Itera sobre cada valor de k da lista
for k in valores_k:
    # Instancia um modelo MiniBatchKMeans com o número de clusters da iteração atual
    modelo = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
    # Executa o agrupamento e obtém os rótulos de cluster
    grupos = modelo.fit_predict(X)
    # Calcula o Silhouette Score da partição atual
    score = silhouette_score(X, grupos, sample_size=10000, random_state=42)
    # Adiciona o resultado à lista
    resultados.append(score)
    # Imprime o resultado intermediário
    print(f"k = {k:2d} -> Silhouette Score = {score:.3f}")

# ==============================================================================
# 3. Plotagem do Gráfico de Sensibilidade
# ==============================================================================

# Cria o gráfico de linha conectando os scores com marcadores circulares
plt.plot(valores_k, resultados, marker="o", color="#1f77b4", linewidth=2)

# Adiciona o rótulo do eixo X
plt.xlabel("Número de clusters (k)", fontsize=11)

# Adiciona o rótulo do eixo Y
plt.ylabel("Silhouette Score", fontsize=11)

# Adiciona o título do gráfico
plt.title("Sensibilidade do Silhouette Score ao Número de Clusters", fontsize=12, fontweight="bold")

# Ativa as linhas de grade para facilitar a leitura visual
plt.grid(True, linestyle="--", alpha=0.6)

# Renderiza a janela com o gráfico
plt.show()
