"""Interface web (Streamlit) do Recomovie AI.

Este arquivo gerencia a camada visual e de interação do usuário:
- Barra de busca com autocomplete integrado.
- Exibição do filme selecionado com detalhes.
- Geração de recomendações via K-Means e Similaridade de Cosseno.
- Busca e renderização de pôsteres e ícones em tempo real na internet (IMDb com fallback para TMDb).
"""

# Importa expressões regulares para extração de links em páginas web
import re

# Importa a biblioteca pandas para manipulação e validação de dados
import pandas as pd

# Importa requests para realizar requisições HTTP para as APIs e CDNs de imagens
import requests

# Importa a biblioteca Streamlit para construção da interface gráfica interativa
import streamlit as st

# Importa o componente st_searchbox para barra de pesquisa com suporte a autocomplete dinâmico
from streamlit_searchbox import st_searchbox

# Importa a classe RecomendadorFilmes do módulo recomendador.py
from recomendador import RecomendadorFilmes


# Define as configurações visuais da página no navegador (título da aba, ícone e layout largo)
st.set_page_config(
    page_title="Recomovie AI - Recomendações de Filmes",
    page_icon="🎬",
    layout="wide"
)

# Define cabeçalho padrão de User-Agent para requisições HTTP simulando um navegador comum
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


# Decorator do Streamlit para manter a instância do modelo em memória e evitar recarregar a cada interação
@st.cache_resource
def carregar_recomendador():
    """Instancia o RecomendadorFilmes uma única vez por sessão do servidor."""
    # Retorna o objeto RecomendadorFilmes devidamente inicializado e treinado
    return RecomendadorFilmes()


# Inicializa ou recupera a instância do recomendador a partir do cache
recomendador = carregar_recomendador()


def _buscar_imdb(imdb_id) -> str:
    """Consulta a API pública de sugestões/metadados da Amazon/IMDb para obter a URL do pôster em alta resolução."""
    # Valida se o imdb_id foi fornecido e não é um valor nulo/NaN
    if not imdb_id or pd.isna(imdb_id):
        return ""
    try:
        # Converte o ID para string, remove casas decimais se houver e retira espaços em branco
        raw_str = str(imdb_id).split(".")[0].strip()
        # Remove o prefixo 'tt' caso já esteja presente na string
        if raw_str.lower().startswith("tt"):
            raw_str = raw_str[2:]
        # Valida se o identificador restante é numérico
        if raw_str.isdigit():
            # Padroniza o ID no formato oficial do IMDb com prefixo 'tt' e 7 dígitos (ex: tt0435761)
            tt_id = f"tt{raw_str.zfill(7)}"
            # Monta a URL do endpoint de metadados do IMDb baseado no primeiro caractere do ID
            endpoint = f"https://v3.sg.media-imdb.com/suggestion/{tt_id[0]}/{tt_id}.json"
            # Realiza a requisição GET com tempo limite de 3 segundos
            resp = requests.get(endpoint, headers=HEADERS, timeout=3)
            # Se a resposta for bem-sucedida (código 200)
            if resp.status_code == 200:
                # Decodifica o JSON retornado pela API
                data = resp.json()
                # Itera sobre a lista de itens encontrados no campo 'd'
                for item in data.get("d", []):
                    # Verifica se o ID do item corresponde exatamente ao ID pesquisado
                    if item.get("id") == tt_id:
                        # Extrai o objeto de imagem 'i'
                        img_info = item.get("i")
                        # Verifica se é um dicionário e contém a chave 'imageUrl'
                        if isinstance(img_info, dict) and "imageUrl" in img_info:
                            img_url = img_info["imageUrl"]
                            # Redimensiona a URL para uma resolução otimizada (UX400) se for do padrão Amazon CDN
                            if "._V1_" in img_url:
                                return f"{img_url.split('._V1_')[0]}._V1_FMjpg_UX400_.jpg"
                            # Retorna a URL original da imagem
                            return img_url
    except Exception:
        # Em caso de erro de conexão ou parsing, segue silenciosamente para não interromper a aplicação
        pass
    # Retorna string vazia caso a imagem não tenha sido encontrada
    return ""


def _buscar_tmdb(tmdb_id) -> str:
    """Fallback: Busca a URL do pôster oficial no portal TMDb através do tmdbId."""
    # Valida se o tmdb_id foi fornecido e não é um valor nulo/NaN
    if not tmdb_id or pd.isna(tmdb_id):
        return ""
    try:
        # Converte o ID para string e retira eventuais decimais e espaços
        raw_str = str(tmdb_id).split(".")[0].strip()
        # Se for um valor numérico válido
        if raw_str.isdigit():
            # Monta a URL da página do filme no portal TMDb
            url = f"https://www.themoviedb.org/movie/{raw_str}"
            # Faz a requisição HTTP GET com timeout de 3 segundos
            resp = requests.get(url, headers=HEADERS, timeout=3)
            # Se o status da página for 200 OK
            if resp.status_code == 200:
                # Procura a tag OpenGraph de imagem (<meta property="og:image" content="...">)
                m = re.search(r'<meta property="og:image" content="(https://[^"]+)"', resp.text)
                if m:
                    # Retorna a URL encontrada no OpenGraph
                    return m.group(1)
                # Procura alternativamente pela tag de classe de pôster
                m2 = re.search(r'class="poster lazyload" data-src="(https://[^"]+)"', resp.text)
                if m2:
                    # Retorna a URL alternativa encontrada
                    return m2.group(1)
    except Exception:
        # Trata exceções retornando vazio
        pass
    # Retorna string vazia se não encontrar
    return ""


# Decorator para armazenar em cache os resultados de imagens buscadas em memória RAM (até 5000 entradas)
@st.cache_data(show_spinner=False, max_entries=5000)
def buscar_poster_tempo_real(imdb_id=None, tmdb_id=None) -> str:
    """Busca o pôster em tempo real: tenta primeiro o IMDb; se não encontrar, tenta o TMDb."""
    # 1. Tenta buscar no provedor principal (IMDb)
    url_imdb = _buscar_imdb(imdb_id)
    # Se encontrou a imagem no IMDb, retorna diretamente
    if url_imdb:
        return url_imdb

    # 2. Se não encontrou no IMDb, executa o fallback no provedor secundário (TMDb)
    url_tmdb = _buscar_tmdb(tmdb_id)
    # Se encontrou a imagem no TMDb, retorna
    if url_tmdb:
        return url_tmdb

    # Retorna string vazia se nenhum dos provedores possuir a imagem
    return ""


def buscar_sugestoes(termo: str):
    """Callback executado pela barra de pesquisa para retornar tuplas (título, movieId) no autocomplete."""
    # Chama o método buscar_filmes da classe do recomendador
    filmes = recomendador.buscar_filmes(termo)
    # Retorna uma lista de tuplas formatadas para o st_searchbox
    return [(filme.title, filme.movieId) for filme in filmes.itertuples()]


# Exibe o título principal da aplicação na interface
st.title("🎬 Recomovie AI")

# Exibe subtítulo com breve instrução
st.write("Escolha um filme e receba recomendações parecidas com pôsteres em tempo real.")

# Exibe caixa informativa com o funcionamento técnico resumido
st.info("O sistema busca os pôsteres diretamente na internet em tempo real (**IMDb / TMDb**) sem arquivos locais.")

# Renderiza a barra única de pesquisa com autocomplete dinâmico
movie_id = st_searchbox(
    buscar_sugestoes,
    placeholder="Digite o nome de um filme (ex: Toy Story, Heat, Matrix, Soul)...",
    label="Buscar filme",
    key="busca_filme",
    clear_on_submit=False,
)

# Verifica se um filme foi selecionado pelo usuário
if movie_id:
    # Obtém todas as informações do filme selecionado a partir do movieId
    filme_selecionado = recomendador.obter_filme(movie_id)

    # Se o filme foi encontrado com sucesso
    if filme_selecionado is not None:
        # Cria duas colunas para o card do filme selecionado: uma para o pôster e outra para o texto
        col_sel_img, col_sel_info = st.columns([1, 6])

        # Coluna da imagem do filme selecionado
        with col_sel_img:
            # Busca a URL do pôster do filme selecionado em tempo real
            sel_poster = buscar_poster_tempo_real(
                filme_selecionado.get("imdbId"),
                filme_selecionado.get("tmdbId")
            )
            # Se a URL foi retornada, exibe a imagem
            if sel_poster:
                st.image(sel_poster, width=105)
            else:
                # Caso contrário, exibe ícone de fallback
                st.markdown("🎬")

        # Coluna de informações do filme selecionado
        with col_sel_info:
            # Exibe o título em destaque
            st.subheader(filme_selecionado['title'])
            # Exibe as métricas de nota, quantidade de avaliadores e cluster
            st.markdown(
                f"**Nota média:** {filme_selecionado['media_nota_r']:.2f} | "
                f"**Avaliadores:** {int(filme_selecionado['quant_avaliadores']):,} | "
                f"**Cluster:** Grupo #{int(filme_selecionado['cluster'])}"
            )

    # Insere uma linha divisória horizontal
    st.write("---")

    # Renderiza o controle deslizante (slider) para escolher o número de recomendações (de 3 a 20, padrão 10)
    quantidade = st.slider("Quantidade de recomendações", 3, 20, 10)

    # Botão com destaque visual para disparar o cálculo das recomendações
    if st.button("Recomendar filmes", type="primary"):
        # Exibe spinner de carregamento durante a execução do cálculo e busca de imagens
        with st.spinner("Buscando recomendações e pôsteres em tempo real..."):
            # Chama o método de recomendação da classe do modelo
            resultado = recomendador.recomendar(movie_id, quantidade)

        # Exibe o título da seção de recomendações
        st.subheader("Filmes recomendados")

        # Verifica se o filme não tem gêneros conhecidos antes de exibir recomendações.
        if resultado.empty:
            generos_disponiveis = [
                genero for genero in recomendador.generos
                if recomendador.dados.loc[recomendador.dados["movieId"] == movie_id, genero].sum() > 0
            ]
            if not generos_disponiveis:
                st.warning(
                    "Este filme não possui gêneros cadastrados. Para evitar recomendações "
                    "enganosas, o sistema não exibirá sugestões para ele."
                )
            else:
                st.warning("Não encontrei recomendações para este filme com os critérios atuais.")
        else:
            # Itera sobre cada filme recomendado retornado
            for _, filme in resultado.iterrows():
                # Busca a imagem correspondente em tempo real
                poster_url = buscar_poster_tempo_real(
                    filme.get("imdbId"),
                    filme.get("tmdbId")
                )

                # Cria duas colunas para o card de recomendação: coluna da esquerda para imagem e direita para texto
                col_poster, col_texto = st.columns([1, 8])

                # Renderiza o pôster na coluna da esquerda
                with col_poster:
                    if poster_url:
                        st.image(poster_url, width=85)
                    else:
                        st.markdown("🎬")

                # Renderiza os dados do filme na coluna da direita
                with col_texto:
                    # Exibe o título do filme em negrito
                    st.markdown(f"**{filme['title']}**")
                    # Exibe a nota média formatada com 2 casas decimais e a similaridade em porcentagem
                    st.markdown(
                        f"Nota média: {filme['media_nota_r']:.2f} | "
                        f"Similaridade: {filme['similaridade']:.1f}%"
                    )

                # Insere uma linha divisória entre cada card de filme
                st.divider()
else:
    # Mensagem exibida enquanto nenhum filme for pesquisado ou selecionado
    st.caption("Comece digitando o nome de um filme para ver sugestões.")

# Rodapé com nota sobre as limitações dos dados
st.caption("Limitação: recomendações dependem dos dados disponíveis e podem não fazer sentido para filmes muito pouco avaliados.")
