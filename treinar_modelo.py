"""Script de linha de comando para retreinar e salvar o modelo do Recomovie AI.

Execução:
    python treinar_modelo.py
"""

# Importa a classe principal do recomendador
from recomendador import RecomendadorFilmes

# Instancia o objeto RecomendadorFilmes (o construtor automaticamente carrega os CSVs e treina o MiniBatchKMeans)
recomendador = RecomendadorFilmes()

# Salva o modelo treinado, o scaler e a lista de features no arquivo 'modelo_recomovie.joblib'
recomendador.salvar()

# Imprime no terminal a mensagem de sucesso
print("Modelo criado e salvo com sucesso: modelo_recomovie.joblib")
