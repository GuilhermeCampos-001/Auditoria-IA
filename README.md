# Recomovie AI

Projeto de recomendação de filmes por agrupamento e similaridade.

## Tecnologias
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Como funciona
1. Os filmes são agrupados com MiniBatchKMeans (configuração final: K=150).
2. As características são 18 gêneros conhecidos e a nota média padronizada.
3. O sistema encontra filmes do mesmo cluster e os ordena pela similaridade de cosseno.
4. Filmes sem gêneros conhecidos não recebem recomendações automáticas, para evitar afinidade enganosa.

## Executar
Instale as dependências e execute a interface:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Auditoria e testes
Na pasta do projeto, execute:

```bash
python auditoria_testes.py
python testes_melhoria.py
python comparar_clusters_ampliado.py
```

- `AUDITORIA/relatorio_auditoria.txt`: resumo final da auditoria.
- `AUDITORIA/relatorio_auditoria_detalhado.txt`: registro detalhado da execução da auditoria.
- `AUDITORIA/resultados_testes_melhoria.txt`: testes de regressão da melhoria.
- `comparacao_clusters_ampliada.csv` e arquivos PNG: comparação de K e métricas.

## Limitações
A coerência de gênero é uma medida básica e não prova que o usuário gostará do filme. O sistema não usa histórico individual de preferências e não analisa diretamente enredo, elenco ou direção. Filmes com poucos dados podem ter recomendações menos confiáveis.
