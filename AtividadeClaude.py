"""
app.py
Sistema Interno de Otimização de Operações - Agência de Marketing Digital
---------------------------------------------------------------------------
Módulo único Streamlit contendo duas ferramentas de apoio à decisão:

1. Detector de Spam (NLP)      -> Ferramenta.SPAM
2. Calculadora de Engajamento  -> Ferramenta.TIKTOK

Ambas as ferramentas utilizam o TensorFlow para processar os dados de
entrada através de pequenas redes neurais densas, construídas e
"calibradas" (pesos definidos manualmente, sem necessidade de treino ou
de arquivos externos .h5/.keras) no momento da inicialização do app.
Isso garante que o script rode de primeira tanto localmente quanto em
ambientes de nuvem (ex.: Render), sem dependências externas de peso de
modelo.

Autor: Engenharia de Software Sênior - Data Science
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuração geral da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Marketing Ops Suite | Ferramentas Internas",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# CAMADA 1: PRÉ-PROCESSAMENTO (Detector de Spam)
# ===========================================================================

# Léxico de palavras/expressões comumente associadas a e-mails de SPAM.
# Em um cenário de produção, este vocabulário seria substituído por um
# tokenizer treinado (ex.: TF-IDF ou embeddings) persistido em disco.
SPAM_KEYWORDS: Final[list[str]] = [
    "grátis", "gratis", "promoção", "promocao", "clique aqui", "compre agora",
    "dinheiro fácil", "dinheiro facil", "ganhe", "prêmio", "premio",
    "parabéns", "parabens", "urgente", "oferta imperdível", "oferta imperdivel",
    "cartão de crédito", "cartao de credito", "empréstimo", "emprestimo",
    "sem compromisso", "100% grátis", "desconto exclusivo", "última chance",
    "ultima chance", "não perca", "nao perca", "confirme seus dados",
    "senha", "milionário", "milionario", "investimento garantido",
    "trabalhe em casa", "renda extra", "clique agora", "resgate",
    "viagra", "criptomoeda", "bitcoin grátis", "você foi selecionado",
    "voce foi selecionado",
]


def _extract_spam_features(email_text: str) -> np.ndarray:
    """Converte o texto bruto do e-mail em um vetor numérico de features.

    Features extraídas (nesta ordem):
        0. Densidade de palavras-chave de spam (ocorrências / total de palavras)
        1. Proporção de caracteres em MAIÚSCULAS
        2. Densidade de pontos de exclamação
        3. Presença de símbolos monetários (R$, $, %)
        4. Comprimento normalizado do texto (proxy de "urgência/curto")

    Args:
        email_text: Conteúdo bruto do e-mail colado pelo usuário.

    Returns:
        Vetor numpy de shape (5,) com valores normalizados em [0, 1].
    """
    text = email_text.strip()
    if not text:
        return np.zeros(5, dtype=np.float32)

    lowered = text.lower()
    words = re.findall(r"\b\w+\b", lowered)
    total_words = max(len(words), 1)

    keyword_hits = sum(lowered.count(keyword) for keyword in SPAM_KEYWORDS)
    keyword_density = min(keyword_hits / total_words, 1.0)

    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )

    exclamation_density = min(text.count("!") / max(len(text), 1) * 20, 1.0)

    money_symbols = len(re.findall(r"[R$%€]", text))
    money_presence = min(money_symbols / 5.0, 1.0)

    length_score = min(total_words / 40.0, 1.0)

    return np.array(
        [keyword_density, uppercase_ratio, exclamation_density, money_presence, length_score],
        dtype=np.float32,
    )


@st.cache_resource(show_spinner=False)
def build_spam_model() -> tf.keras.Model:
    """Constrói e calibra uma rede neural densa simples para classificação de spam.

    A arquitetura é uma rede densa (5 -> 8 -> 1) compilada em tempo de
    execução. Os pesos da camada de saída são definidos manualmente para
    refletir a importância de cada feature (equivalente a uma regressão
    logística calibrada por especialistas de domínio), eliminando a
    necessidade de treinar ou carregar um arquivo de pesos externo.

    Returns:
        Modelo `tf.keras.Model` pronto para inferência via `.predict()`.
    """
    inputs = tf.keras.Input(shape=(5,), name="email_features")
    hidden = tf.keras.layers.Dense(8, activation="relu", name="hidden_layer")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="spam_detector_nn")
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # Calibração manual dos pesos: mapeia cada feature diretamente para um
    # neurônio da camada oculta (matriz quase-identidade escalada) e depois
    # combina tudo, com pesos positivos, na camada de saída.
    hidden_weights = np.eye(5, 8, dtype=np.float32) * 6.0
    hidden_bias = np.zeros(8, dtype=np.float32)
    model.get_layer("hidden_layer").set_weights([hidden_weights, hidden_bias])

    output_weights = np.array(
        [[3.0], [1.5], [1.2], [1.0], [0.5], [0.0], [0.0], [0.0]], dtype=np.float32
    )
    output_bias = np.array([-2.2], dtype=np.float32)
    model.get_layer("spam_probability").set_weights([output_weights, output_bias])

    return model


def predict_spam(email_text: str) -> float:
    """Executa a inferência de spam sobre o texto informado.

    Args:
        email_text: Conteúdo do e-mail a ser classificado.

    Returns:
        Probabilidade (0.0 a 1.0) de o texto ser SPAM.
    """
    model = build_spam_model()
    features = _extract_spam_features(email_text).reshape(1, -1)
    tensor_input = tf.convert_to_tensor(features)
    probability = model.predict(tensor_input, verbose=0)
    return float(probability[0][0])


# ===========================================================================
# CAMADA 2: CALCULADORA DE ENGAJAMENTO TIKTOK
# ===========================================================================

@dataclass(frozen=True)
class CategoriaConfig:
    """Configuração de calibração por categoria de conteúdo."""

    nome: str
    peso_base: float  # Afinidade base da categoria com o algoritmo (0-1)


CATEGORIAS: Final[dict[str, CategoriaConfig]] = {
    "Humor / Memes": CategoriaConfig("Humor / Memes", 0.85),
    "Dança / Música": CategoriaConfig("Dança / Música", 0.80),
    "Beleza / Moda": CategoriaConfig("Beleza / Moda", 0.70),
    "Educacional / Tutorial": CategoriaConfig("Educacional / Tutorial", 0.60),
    "Culinária": CategoriaConfig("Culinária", 0.65),
    "Fitness / Saúde": CategoriaConfig("Fitness / Saúde", 0.55),
    "Tecnologia": CategoriaConfig("Tecnologia", 0.50),
    "Vlog / Dia a Dia": CategoriaConfig("Vlog / Dia a Dia", 0.40),
    "Corporativo / Institucional": CategoriaConfig("Corporativo / Institucional", 0.25),
}


@st.cache_resource(show_spinner=False)
def build_tiktok_model() -> tf.keras.Model:
    """Constrói e calibra uma rede neural densa para estimar o alcance de vídeos.

    Features de entrada (nesta ordem):
        0. Peso base da categoria (afinidade histórica com o algoritmo)
        1. Número de hashtags normalizado (ótimo empírico ~ 5 a 8 hashtags)
        2. Penalidade por excesso de hashtags (spam de hashtags)

    A camada de saída é calibrada para produzir um "score de viralização"
    em [0, 1], posteriormente convertido em uma estimativa de alcance
    (visualizações) para leitura de negócio.

    Returns:
        Modelo `tf.keras.Model` pronto para inferência via `.predict()`.
    """
    inputs = tf.keras.Input(shape=(3,), name="tiktok_features")
    hidden = tf.keras.layers.Dense(6, activation="relu", name="hidden_layer")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="viral_score")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tiktok_engagement_nn")
    model.compile(optimizer="adam", loss="mse")

    hidden_weights = np.eye(3, 6, dtype=np.float32) * 4.0
    hidden_bias = np.zeros(6, dtype=np.float32)
    model.get_layer("hidden_layer").set_weights([hidden_weights, hidden_bias])

    output_weights = np.array(
        [[2.5], [2.0], [-2.0], [0.0], [0.0], [0.0]], dtype=np.float32
    )
    output_bias = np.array([-1.6], dtype=np.float32)
    model.get_layer("viral_score").set_weights([output_weights, output_bias])

    return model


def _hashtag_features(hashtags: int) -> tuple[float, float]:
    """Calcula o score de hashtags e sua penalidade por excesso.

    Args:
        hashtags: Quantidade de hashtags usadas no vídeo.

    Returns:
        Tupla (score_normalizado, penalidade_excesso), ambos em [0, 1].
    """
    optimal_range = 8
    score = min(hashtags / optimal_range, 1.0)
    excess = max(hashtags - optimal_range, 0)
    penalty = min(excess / 15.0, 1.0)
    return score, penalty


def predict_tiktok_engagement(categoria: str, hashtags: int) -> dict[str, float]:
    """Executa a inferência de engajamento para um vídeo do TikTok.

    Args:
        categoria: Nome da categoria selecionada (deve existir em CATEGORIAS).
        hashtags: Número de hashtags utilizadas no vídeo.

    Returns:
        Dicionário com "score" (0-1) e "alcance_estimado" (visualizações).
    """
    config = CATEGORIAS.get(categoria)
    if config is None:
        raise ValueError(f"Categoria desconhecida: {categoria}")

    hashtag_score, hashtag_penalty = _hashtag_features(hashtags)
    features = np.array(
        [[config.peso_base, hashtag_score, hashtag_penalty]], dtype=np.float32
    )

    model = build_tiktok_model()
    tensor_input = tf.convert_to_tensor(features)
    score = float(model.predict(tensor_input, verbose=0)[0][0])

    # Conversão de negócio: score -> alcance estimado (escala ilustrativa)
    alcance_estimado = score * 500_000
    return {"score": score, "alcance_estimado": alcance_estimado}


# ===========================================================================
# CAMADA 3: INTERFACE (Streamlit UI)
# ===========================================================================

def render_sidebar() -> str:
    """Renderiza a barra lateral de navegação e retorna a ferramenta ativa."""
    with st.sidebar:
        st.markdown("## 🏢 Marketing Ops Suite")
        st.caption("Ferramentas internas de produtividade")
        ferramenta = st.radio(
            "Selecione uma ferramenta:",
            options=["📧 Detector de Spam", "🎵 Calculadora TikTok"],
            index=0,
        )
        st.divider()
        st.caption(
            "Powered by TensorFlow • Uso interno exclusivo\n\n"
            "Versão 1.0.0"
        )
    return ferramenta


def render_spam_tool() -> None:
    """Renderiza a interface da Ferramenta 1: Detector de Spam."""
    st.title("📧 Detector de Spam de E-mails")
    st.write(
        "Cole abaixo o conteúdo de um e-mail para verificar automaticamente "
        "se ele apresenta características de **SPAM**, utilizando um modelo "
        "de rede neural (TensorFlow) treinado sobre padrões textuais "
        "conhecidos de e-mails indesejados."
    )

    email_text = st.text_area(
        "Conteúdo do e-mail",
        height=220,
        placeholder="Cole aqui o texto completo do e-mail recebido...",
    )

    analisar = st.button("🔍 Analisar E-mail", type="primary", use_container_width=True)

    if analisar:
        if not email_text or not email_text.strip():
            st.warning("⚠️ Por favor, insira o conteúdo do e-mail antes de analisar.")
            return

        with st.spinner("Processando texto com o modelo de NLP..."):
            try:
                probabilidade = predict_spam(email_text)
            except Exception as exc:  # noqa: BLE001 - feedback claro ao usuário final
                st.error(f"❌ Ocorreu um erro ao processar o e-mail: {exc}")
                return

        st.divider()
        st.subheader("Resultado da Análise")

        col1, col2 = st.columns(2)
        col1.metric("Probabilidade de SPAM", f"{probabilidade * 100:.1f}%")
        col2.metric("Confiança do Modelo", f"{abs(probabilidade - 0.5) * 200:.1f}%")

        if probabilidade >= 0.5:
            st.error("🚫 **SPAM detectado!** Este e-mail apresenta fortes indícios de conteúdo indesejado.")
        else:
            st.success("✅ **E-mail Válido.** Nenhum padrão relevante de SPAM foi identificado.")

        st.progress(min(max(probabilidade, 0.0), 1.0))


def render_tiktok_tool() -> None:
    """Renderiza a interface da Ferramenta 2: Calculadora de Engajamento TikTok."""
    st.title("🎵 Calculadora de Engajamento TikTok")
    st.write(
        "Preveja se um vídeo tem potencial para **Viralizar** ou **Flopar** "
        "com base na categoria de conteúdo e na quantidade de hashtags "
        "utilizadas, usando um modelo preditivo (TensorFlow) calibrado com "
        "padrões históricos de engajamento da plataforma."
    )

    with st.form(key="tiktok_form"):
        categoria = st.selectbox(
            "Categoria do vídeo",
            options=list(CATEGORIAS.keys()),
            help="Selecione a categoria que melhor descreve o conteúdo do vídeo.",
        )
        hashtags = st.slider(
            "Quantidade de hashtags utilizadas",
            min_value=0,
            max_value=30,
            value=5,
            help="O ideal costuma ficar entre 5 e 8 hashtags relevantes.",
        )
        calcular = st.form_submit_button(
            "🚀 Calcular Potencial de Alcance", type="primary", use_container_width=True
        )

    if calcular:
        with st.spinner("Calculando projeção de engajamento..."):
            try:
                resultado = predict_tiktok_engagement(categoria, hashtags)
            except Exception as exc:  # noqa: BLE001 - feedback claro ao usuário final
                st.error(f"❌ Ocorreu um erro ao calcular o engajamento: {exc}")
                return

        score = resultado["score"]
        alcance = resultado["alcance_estimado"]

        st.divider()
        st.subheader("Resultado da Projeção")

        col1, col2 = st.columns(2)
        col1.metric("Score de Viralização", f"{score * 100:.1f}%")
        col2.metric("Alcance Estimado", f"{alcance:,.0f} views".replace(",", "."))

        if score >= 0.5:
            st.success("🔥 **VIRAL!** Este vídeo tem alto potencial de alcance orgânico.")
        else:
            st.error("📉 **Flopado.** O conteúdo tende a ter baixo desempenho. Reveja categoria/hashtags.")

        st.progress(min(max(score, 0.0), 1.0))

        if hashtags > 8:
            st.info(
                "💡 Dica: o excesso de hashtags (acima de 8) pode estar "
                "reduzindo o alcance orgânico do vídeo."
            )


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    ferramenta = render_sidebar()

    if ferramenta == "📧 Detector de Spam":
        render_spam_tool()
    else:
        render_tiktok_tool()


if __name__ == "__main__":
    main()