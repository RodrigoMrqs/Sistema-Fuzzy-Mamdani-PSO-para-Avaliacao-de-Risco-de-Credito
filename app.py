"""
Sistema Fuzzy Mamdani + PSO — Avaliação de Risco de Crédito
Aplicação web interativa (Streamlit)
Disciplina: Inteligência Artificial e Computacional (0700M8) — CESUPA 01/2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skfuzzy as fuzz
import json
import os
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risco de Crédito — Fuzzy + PSO",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4e9af1;
        margin-bottom: 0.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy system — default parameters
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "renda_muito_baixa": [0, 0, 4000],
    "renda_baixa":       [1000, 5000, 9000],
    "renda_media":       [5000, 10000, 15000],
    "renda_alta":        [10000, 15000, 19000],
    "renda_muito_alta":  [15000, 20000, 20000],
    "comp_baixo":        [0, 0, 25],
    "comp_moderado":     [10, 30, 50],
    "comp_alto":         [40, 60, 80],
    "comp_muito_alto":   [65, 100, 100],
    "score_muito_ruim":  [0, 0, 200],
    "score_ruim":        [100, 250, 400],
    "score_regular":     [300, 500, 700],
    "score_bom":         [550, 700, 850],
    "score_muito_bom":   [750, 1000, 1000],
    "tempo_curto":       [0, 0, 5],
    "tempo_moderado":    [2, 7, 14],
    "tempo_longo":       [9, 16, 23],
    "tempo_muito_longo": [18, 30, 30],
    "risco_muito_baixo": [0, 0, 20],
    "risco_baixo":       [10, 25, 40],
    "risco_medio":       [30, 50, 70],
    "risco_alto":        [60, 75, 90],
    "risco_muito_alto":  [80, 100, 100],
}

U_RISCO = np.linspace(0, 100, 501)

LABELS_PT = {
    "muito_baixa": "Muito Baixa", "baixa": "Baixa", "media": "Média",
    "alta": "Alta", "muito_alta": "Muito Alta",
    "baixo": "Baixo", "moderado": "Moderado", "alto": "Alto", "muito_alto": "Muito Alto",
    "muito_ruim": "Muito Ruim", "ruim": "Ruim", "regular": "Regular",
    "bom": "Bom", "muito_bom": "Muito Bom",
    "curto": "Curto", "longo": "Longo", "muito_longo": "Muito Longo",
    "muito_baixo": "Muito Baixo", "medio": "Médio",
}

RISK_ZONES = [
    (0,  15, "#1a9641", "Muito Baixo"),
    (15, 35, "#52b947", "Baixo"),
    (35, 65, "#e8a820", "Médio"),
    (65, 85, "#e05c1a", "Alto"),
    (85, 100, "#d73027", "Muito Alto"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Load optimized parameters
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_params():
    path = Path("resultados/fuzzy_depois/params_otimizados.json")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: [float(v) for v in vals] for k, vals in raw.items()}, True
    return DEFAULT_PARAMS, False


PARAMS, OPTIMIZED = load_params()

# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy inference engine
# ─────────────────────────────────────────────────────────────────────────────
def trimf_val(x, a, b, c):
    if x == b:
        return 1.0
    if a < b and a < x < b:
        return (x - a) / (b - a)
    if b < c and b < x < c:
        return (c - x) / (c - b)
    return 0.0


def mamdani_infer(renda_v, comp_v, score_v, tempo_v, params=None):
    """Mamdani inference → (risk_score, active_rules, aggregated_output)."""
    p = params if params is not None else PARAMS

    mr = {k: trimf_val(renda_v, *p[f"renda_{k}"])
          for k in ["muito_baixa", "baixa", "media", "alta", "muito_alta"]}
    mc = {k: trimf_val(comp_v,  *p[f"comp_{k}"])
          for k in ["baixo", "moderado", "alto", "muito_alto"]}
    ms = {k: trimf_val(score_v, *p[f"score_{k}"])
          for k in ["muito_ruim", "ruim", "regular", "bom", "muito_bom"]}
    mt = {k: trimf_val(tempo_v, *p[f"tempo_{k}"])
          for k in ["curto", "moderado", "longo", "muito_longo"]}

    RULES = [
        (min(ms["muito_bom"],  mc["baixo"]),                        "muito_baixo", "score Muito Bom ∧ comp Baixo"),
        (min(ms["muito_bom"],  mc["moderado"]),                     "baixo",       "score Muito Bom ∧ comp Moderado"),
        (min(ms["muito_bom"],  mc["alto"]),                         "medio",       "score Muito Bom ∧ comp Alto"),
        (min(ms["bom"],        mc["baixo"]),                        "muito_baixo", "score Bom ∧ comp Baixo"),
        (min(ms["bom"],        mc["moderado"]),                     "baixo",       "score Bom ∧ comp Moderado"),
        (min(ms["bom"],        mc["alto"]),                         "medio",       "score Bom ∧ comp Alto"),
        (min(ms["regular"],    mc["baixo"]),                        "baixo",       "score Regular ∧ comp Baixo"),
        (min(ms["regular"],    mc["moderado"]),                     "medio",       "score Regular ∧ comp Moderado"),
        (min(ms["regular"],    mc["alto"]),                         "alto",        "score Regular ∧ comp Alto"),
        (min(ms["ruim"],       mc["baixo"]),                        "medio",       "score Ruim ∧ comp Baixo"),
        (min(ms["ruim"],       mc["moderado"]),                     "alto",        "score Ruim ∧ comp Moderado"),
        (min(ms["ruim"],       mc["alto"]),                         "muito_alto",  "score Ruim ∧ comp Alto"),
        (ms["muito_ruim"],                                           "muito_alto",  "score Muito Ruim"),
        (min(mr["muito_alta"], ms["bom"],     mc["baixo"]),         "muito_baixo", "renda Muito Alta ∧ score Bom ∧ comp Baixo"),
        (min(mr["muito_baixa"],mc["muito_alto"]),                   "muito_alto",  "renda Muito Baixa ∧ comp Muito Alto"),
        (min(ms["regular"],    mt["muito_longo"], mc["baixo"]),     "baixo",       "score Regular ∧ tempo Muito Longo ∧ comp Baixo"),
        (min(ms["bom"],        mt["muito_longo"]),                  "muito_baixo", "score Bom ∧ tempo Muito Longo"),
        (min(ms["regular"],    mt["curto"],   mr["baixa"]),         "alto",        "score Regular ∧ tempo Curto ∧ renda Baixa"),
    ]

    agg = np.zeros_like(U_RISCO)
    active = []
    for forca, termo, desc in RULES:
        if forca > 0.001:
            active.append((forca, termo, desc))
        if forca > 0:
            agg = np.maximum(agg, np.minimum(forca, fuzz.trimf(U_RISCO, p[f"risco_{termo}"])))

    total = np.trapz(agg, U_RISCO)
    result = float(np.trapz(agg * U_RISCO, U_RISCO) / total) if total > 1e-10 else 50.0
    active.sort(key=lambda x: -x[0])
    return result, active, agg


def risk_label_info(score):
    for lo, hi, color, label in RISK_ZONES:
        if lo <= score < hi or (score == 100 and hi == 100):
            return label, color
    return "Muito Alto", "#d73027"


def dominant_term(value, variable):
    """Return the term with highest membership for the given value."""
    terms = {
        "renda": ["muito_baixa", "baixa", "media", "alta", "muito_alta"],
        "comp":  ["baixo", "moderado", "alto", "muito_alto"],
        "score": ["muito_ruim", "ruim", "regular", "bom", "muito_bom"],
        "tempo": ["curto", "moderado", "longo", "muito_longo"],
    }[variable]
    best, best_mu = terms[0], 0.0
    for t in terms:
        mu = trimf_val(value, *PARAMS[f"{variable}_{t}"])
        if mu > best_mu:
            best_mu = mu
            best = t
    return LABELS_PT.get(best, best), round(best_mu, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gauge(score):
    """Horizontal gauge chart."""
    fig, ax = plt.subplots(figsize=(9, 2.2))
    for lo, hi, color, lbl in RISK_ZONES:
        ax.barh(0, hi - lo, left=lo, height=0.7, color=color, alpha=0.88, zorder=1)
        ax.text((lo + hi) / 2, 0, lbl, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold", zorder=3)
    # Needle
    ax.plot([score, score], [-0.52, 0.52], color="white", lw=3, zorder=5)
    ax.plot(score, 0.52, marker="v", color="white", markersize=11, zorder=6)
    ax.text(score, -0.72, f"{score:.1f}", ha="center", va="top",
            fontsize=14, color="white", fontweight="bold")
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1.0, 0.9)
    ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout(pad=0.2)
    return fig


def make_output_dist(agg):
    """Plot aggregated output distribution + centroid."""
    score = float(np.trapz(agg * U_RISCO, U_RISCO) / np.trapz(agg, U_RISCO)) if np.trapz(agg, U_RISCO) > 1e-10 else 50.0
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d24")
    ax.fill_between(U_RISCO, agg, alpha=0.45, color="#4e9af1")
    ax.plot(U_RISCO, agg, color="#4e9af1", lw=1.8)
    ax.axvline(score, color="#f5a623", lw=2.2, ls="--", label=f"Centroide = {score:.1f}")
    ax.set_xlabel("Risco [0–100]", color="#aaa", fontsize=9)
    ax.set_ylabel("Pertinência agregada", color="#aaa", fontsize=9)
    ax.set_title("Distribuição de saída (pós-agregação)", color="white", fontsize=10)
    ax.tick_params(colors="#666", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.legend(fontsize=9, facecolor="#1a1d24", edgecolor="#333", labelcolor="white")
    plt.tight_layout()
    return fig


def make_mf_activations(renda_v, comp_v, score_v, tempo_v):
    """4-panel membership function activation chart."""
    PALETTE = ["#4e9af1", "#f1914e", "#4ef17d", "#f14e4e", "#c44ef1"]
    VARS = [
        ("Renda (R$)",       "renda",  renda_v,  np.linspace(0, 20000, 501),
         ["muito_baixa", "baixa", "media", "alta", "muito_alta"]),
        ("Comprometimento (%)", "comp", comp_v, np.linspace(0, 100, 501),
         ["baixo", "moderado", "alto", "muito_alto"]),
        ("Score (pts)",      "score",  score_v,  np.linspace(0, 1000, 501),
         ["muito_ruim", "ruim", "regular", "bom", "muito_bom"]),
        ("Tempo Emprego (anos)", "tempo", tempo_v, np.linspace(0, 30, 501),
         ["curto", "moderado", "longo", "muito_longo"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    fig.patch.set_facecolor("#0e1117")

    for ax, (title, var, val, univ, terms) in zip(axes, VARS):
        ax.set_facecolor("#1a1d24")
        for j, term in enumerate(terms):
            mf_vals = fuzz.trimf(univ, PARAMS[f"{var}_{term}"])
            mu = trimf_val(val, *PARAMS[f"{var}_{term}"])
            active = mu > 0.005
            ax.plot(univ, mf_vals, color=PALETTE[j % len(PALETTE)],
                    lw=2.5 if active else 1.0, alpha=0.9 if active else 0.3,
                    label=f"{LABELS_PT.get(term, term)} ({mu:.2f})" if active else LABELS_PT.get(term, term))
            if active:
                ax.fill_between(univ, mf_vals, alpha=0.12, color=PALETTE[j % len(PALETTE)])
        ax.axvline(val, color="white", ls="--", lw=1.8, alpha=0.85)
        ax.set_title(title, color="white", fontsize=9.5)
        ax.tick_params(colors="#666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.legend(fontsize=6.5, facecolor="#1a1d24", edgecolor="#333",
                  labelcolor="white", loc="upper right")

    plt.tight_layout(pad=0.6)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# App header
# ─────────────────────────────────────────────────────────────────────────────
col_hdr, col_badge = st.columns([4, 1])
with col_hdr:
    st.markdown("## 💳 Avaliação de Risco de Crédito")
    st.caption(
        "Sistema Fuzzy Mamdani + PSO · Inteligência Artificial e Computacional (0700M8) · CESUPA 01/2026"
    )
with col_badge:
    if OPTIMIZED:
        st.success("Parâmetros otimizados por PSO", icon="✅")
    else:
        st.warning("Parâmetros padrão\n(execute o notebook)", icon="⚠️")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Navigation tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_avaliacao, tab_resultados, tab_metodologia = st.tabs([
    "🏦  Avaliação de Risco",
    "📊  Resultados Experimentais",
    "📋  Metodologia",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive Assessment
# ═══════════════════════════════════════════════════════════════════════════════
with tab_avaliacao:
    col_form, col_result = st.columns([1, 1.7], gap="large")

    # ── Inputs ──────────────────────────────────────────────────────────────
    with col_form:
        st.subheader("Dados do Solicitante")

        renda = st.slider(
            "Renda Mensal Bruta (R$)",
            0, 20000, 6000, 100,
            help="Renda mensal bruta declarada pelo solicitante",
        )
        rl, rmu = dominant_term(renda, "renda")
        st.caption(f"Classe predominante: **{rl}** (μ = {rmu})")

        comp = st.slider(
            "Comprometimento da Renda com Dívidas (%)",
            0, 100, 30, 1,
            help="Percentual da renda já comprometida com outras obrigações financeiras",
        )
        cl, cmu = dominant_term(comp, "comp")
        st.caption(f"Classe predominante: **{cl}** (μ = {cmu})")

        score_cr = st.slider(
            "Score de Crédito (0 – 1000)",
            0, 1000, 650, 5,
            help="Pontuação de crédito do solicitante (ex: Serasa Score)",
        )
        sl, smu = dominant_term(score_cr, "score")
        st.caption(f"Classe predominante: **{sl}** (μ = {smu})")

        tempo = st.slider(
            "Tempo no Emprego Atual (anos)",
            0, 30, 5, 1,
            help="Anos de vínculo empregatício atual",
        )
        tl, tmu = dominant_term(tempo, "tempo")
        st.caption(f"Classe predominante: **{tl}** (μ = {tmu})")

        st.divider()
        st.caption("O resultado é atualizado automaticamente conforme você ajusta os controles.")

    # ── Result ───────────────────────────────────────────────────────────────
    with col_result:
        risk_score, active_rules, agg_output = mamdani_infer(renda, comp, score_cr, tempo)
        label, color = risk_label_info(risk_score)

        st.subheader("Resultado da Análise")

        m1, m2, m3 = st.columns(3)
        m1.metric("Índice de Risco", f"{risk_score:.1f} / 100")
        m2.metric("Classificação", label)
        m3.metric("Regras ativadas", f"{len(active_rules)} / 18")

        st.pyplot(make_gauge(risk_score), use_container_width=True)

        # Decision explanation
        col_rec, col_dist = st.columns([1, 1])
        with col_rec:
            if risk_score < 35:
                st.success(
                    f"**Recomendação: Aprovar**\n\n"
                    f"Risco {label.lower()} — perfil financeiro favorável.",
                    icon="✅",
                )
            elif risk_score < 65:
                st.warning(
                    f"**Recomendação: Analisar**\n\n"
                    f"Risco médio — requer análise complementar.",
                    icon="⚠️",
                )
            else:
                st.error(
                    f"**Recomendação: Recusar / Garantias adicionais**\n\n"
                    f"Risco {label.lower()} — perfil de alto risco.",
                    icon="🚫",
                )
        with col_dist:
            st.pyplot(make_output_dist(agg_output), use_container_width=True)

    # ── Active rules ─────────────────────────────────────────────────────────
    st.divider()
    col_rules, col_mfs = st.columns([1, 2])

    with col_rules:
        with st.expander(f"📜 Regras Fuzzy Ativadas ({len(active_rules)} / 18)", expanded=True):
            if active_rules:
                RISK_COLORS_PT = {
                    "muito_baixo": "🟢", "baixo": "🟢",
                    "medio": "🟡", "alto": "🟠", "muito_alto": "🔴",
                }
                for forca, termo, desc in active_rules:
                    icon = RISK_COLORS_PT.get(termo, "⚪")
                    st.markdown(
                        f"{icon} **SE** {desc}  \n"
                        f"→ **Risco {LABELS_PT.get(termo, termo)}** *(força: {forca:.3f})*"
                    )
            else:
                st.info("Nenhuma regra ativada com força > 0,5%.")

    with col_mfs:
        st.markdown("**Ativação das Funções de Pertinência**")
        st.caption("Linha tracejada = valor do solicitante · MFs em destaque = mais ativas")
        st.pyplot(make_mf_activations(renda, comp, score_cr, tempo), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Experimental Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_resultados:
    st.subheader("Desempenho da Otimização PSO")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE — Padrão",  "5,36")
    m2.metric("MAE — PSO (média)", "~0,51", delta="−90,5 %", delta_color="inverse")
    m3.metric("Melhor run (5 exec.)", "~0,20", delta="−96,3 %", delta_color="inverse")
    m4.metric("Partículas / Iterações", "40 / 150")
    m5.metric("Dimensão do espaço", "69 parâmetros")

    st.divider()

    # Comparison table
    csv_path = Path("resultados/comparacao_cenarios.csv")
    if csv_path.exists():
        st.subheader("Comparação por Cenário (10 casos)")
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()
        # Detect column names robustly (notebook versions differ)
        col_ea = next((c for c in cols if c.lower() in ("ea", "erro_antes", "erro antes", "error_before")), None)
        col_ed = next((c for c in cols if c.lower() in ("ed", "erro_depois", "erro depois", "error_after")), None)
        if col_ea and col_ed:
            df["Melhoria"] = ((df[col_ea] - df[col_ed]) / df[col_ea].replace(0, 1) * 100).round(1)
            df["Melhoria"] = df["Melhoria"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(
                df.style.background_gradient(subset=[col_ea, col_ed], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Execute o notebook para gerar a tabela de cenários.", icon="ℹ️")

    st.divider()

    # Convergence plots
    st.subheader("Curvas de Convergência")
    paths_conv = [
        ("resultados/evolucao/convergencia_pso.png",   "Convergência PSO — run principal (seed=42)"),
        ("resultados/evolucao/convergencia_5runs.png",  "Convergência — 5 execuções independentes"),
        ("resultados/evolucao/comparacao_pso_rs.png",   "PSO vs Busca Aleatória — mesmo orçamento computacional"),
    ]
    for path, caption in paths_conv:
        if Path(path).exists():
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"Execute o notebook para gerar: `{path}`", icon="ℹ️")

    st.divider()

    # MF plots
    st.subheader("Funções de Pertinência")
    col_before, col_after = st.columns(2)
    for col, (path, cap) in zip(
        [col_before, col_after],
        [
            ("resultados/fuzzy_antes/funcoes_pertinencia.png", "MFs — Parâmetros Padrão"),
            ("resultados/fuzzy_depois/mfs_otimizadas.png",     "MFs — Após Otimização PSO"),
        ],
    ):
        if Path(path).exists():
            col.image(path, caption=cap, use_container_width=True)
        else:
            col.info(f"Execute o notebook: `{path}`", icon="ℹ️")

    surf_path = "resultados/fuzzy_antes/superficie_controle.png"
    if Path(surf_path).exists():
        st.image(surf_path, caption="Superfície de Controle — Risco(Score, Comprometimento)",
                 use_container_width=True)

    impact_path = "resultados/fuzzy_depois/impacto_mfs.png"
    if Path(impact_path).exists():
        st.subheader("Análise de Impacto — Deslocamento das MFs após PSO")
        st.image(impact_path, caption="Top 15 MFs com maior deslocamento de pico (relativo ao universo da variável)",
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metodologia:
    col_v, col_pso = st.columns(2)

    with col_v:
        st.subheader("Variáveis do Sistema")
        vars_df = pd.DataFrame([
            ["Renda Mensal",        "Entrada", "[0 – 20 000] R$",    "5 termos"],
            ["Comprometimento",     "Entrada", "[0 – 100] %",         "4 termos"],
            ["Score de Crédito",    "Entrada", "[0 – 1 000] pts",     "5 termos"],
            ["Tempo no Emprego",    "Entrada", "[0 – 30] anos",       "4 termos"],
            ["Risco de Crédito",    "Saída",   "[0 – 100]",           "5 termos"],
        ], columns=["Variável", "Tipo", "Universo", "MFs"])
        st.dataframe(vars_df, hide_index=True, use_container_width=True)

        st.markdown("""
        **Inferência:** Mamdani (min-implicação, max-agregação)
        **Defuzzificação:** Centroide
        **Formato das MFs:** Triangular `[a, b, c]`
        """)

    with col_pso:
        st.subheader("Parâmetros do PSO")
        pso_df = pd.DataFrame([
            ["Partículas (n)",               "40"],
            ["Iterações máximas (T)",         "150"],
            ["Inércia (w)",                   "0,7"],
            ["Componente cognitiva (c₁)",     "1,5"],
            ["Componente social (c₂)",        "1,5"],
            ["Dimensão do espaço",            "69 (23 MFs × 3 params)"],
            ["Função objetivo",               "MAE — 10 cenários"],
            ["Execuções independentes",       "5 (seeds: 0, 7, 13, 21, 42)"],
            ["Orçamento por execução",        "6 040 avaliações"],
        ], columns=["Parâmetro", "Valor"])
        st.dataframe(pso_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Base de Regras Fuzzy (18 regras)")

    RULE_COLORS = {
        "Muito Baixo": ":green[**Muito Baixo**]",
        "Baixo":       ":green[**Baixo**]",
        "Médio":       ":orange[**Médio**]",
        "Alto":        "🔴 **Alto**",
        "Muito Alto":  "🔴 **Muito Alto**",
    }

    rules_df = pd.DataFrame([
        [1,  "score Muito Bom ∧ comp Baixo",                              "Muito Baixo"],
        [2,  "score Muito Bom ∧ comp Moderado",                           "Baixo"],
        [3,  "score Muito Bom ∧ comp Alto",                               "Médio"],
        [4,  "score Bom ∧ comp Baixo",                                    "Muito Baixo"],
        [5,  "score Bom ∧ comp Moderado",                                 "Baixo"],
        [6,  "score Bom ∧ comp Alto",                                     "Médio"],
        [7,  "score Regular ∧ comp Baixo",                                "Baixo"],
        [8,  "score Regular ∧ comp Moderado",                             "Médio"],
        [9,  "score Regular ∧ comp Alto",                                 "Alto"],
        [10, "score Ruim ∧ comp Baixo",                                   "Médio"],
        [11, "score Ruim ∧ comp Moderado",                                "Alto"],
        [12, "score Ruim ∧ comp Alto",                                    "Muito Alto"],
        [13, "score Muito Ruim",                                           "Muito Alto"],
        [14, "renda Muito Alta ∧ score Bom ∧ comp Baixo",                "Muito Baixo"],
        [15, "renda Muito Baixa ∧ comp Muito Alto",                       "Muito Alto"],
        [16, "score Regular ∧ tempo Muito Longo ∧ comp Baixo",           "Baixo"],
        [17, "score Bom ∧ tempo Muito Longo",                             "Muito Baixo"],
        [18, "score Regular ∧ tempo Curto ∧ renda Baixa",                "Alto"],
    ], columns=["#", "SE (Antecedente)", "Consequente (Risco)"])

    def _style_consequente(val):
        cmap = {
            "Muito Baixo": "background-color:#1a9641;color:white",
            "Baixo":       "background-color:#52b947;color:white",
            "Médio":       "background-color:#e8a820;color:black",
            "Alto":        "background-color:#e05c1a;color:white",
            "Muito Alto":  "background-color:#d73027;color:white",
        }
        return cmap.get(val, "")

    st.dataframe(
        rules_df.style.map(_style_consequente, subset=["Consequente (Risco)"]),
        hide_index=True, use_container_width=True,
    )

    st.divider()
    st.subheader("Equações do PSO (Kennedy & Eberhart, 1995)")
    st.code(
        "v[t+1] = w · v[t]\n"
        "       + c1 · r1 · (pbest − x[t])    ← aprendizado individual\n"
        "       + c2 · r2 · (gbest − x[t])    ← aprendizado social\n"
        "\n"
        "x[t+1] = clip(x[t] + v[t+1], lb, ub)",
        language="text",
    )

    st.info(
        "**Alternativa 2 — Integração Fuzzy-Evolutiva:**  \n"
        "O sistema Fuzzy Mamdani **é** a função de aptidão do PSO.  \n"
        "Cada partícula representa uma configuração completa das 23 MFs (69 parâmetros reais).  \n"
        "O PSO minimiza o MAE do sistema fuzzy sobre 10 cenários de referência.",
        icon="🔗",
    )
