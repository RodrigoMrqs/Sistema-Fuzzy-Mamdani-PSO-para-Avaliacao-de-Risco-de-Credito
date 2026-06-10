# Sistema Fuzzy Mamdani + PSO — Avaliação de Risco de Crédito

**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA 01/2026  
**AV2 Parte 1:** Sistema de Controle Fuzzy (Opção B — Produto)  
**AV2 Parte 2:** Computação Bioinspirada — PSO (Alternativa 2: Integração Fuzzy-Evolutiva)

---

## Descrição

Sistema de apoio à decisão que avalia o **risco de crédito** de um solicitante de empréstimo usando lógica fuzzy Mamdani. Os parâmetros das funções de pertinência são otimizados automaticamente pelo algoritmo **PSO (Particle Swarm Optimization)**, que usa o próprio sistema fuzzy como função de aptidão.

O notebook compara o PSO com **Busca Aleatória** (mesmo orçamento computacional) para evidenciar a eficiência do mecanismo de aprendizado coletivo do PSO.

### Variáveis de entrada

| Variável | Universo | Termos linguísticos |
|----------|----------|---------------------|
| `renda` | [0 – 20 000] R$ | Muito Baixa, Baixa, Média, Alta, Muito Alta |
| `comprometimento` | [0 – 100] % | Baixo, Moderado, Alto, Muito Alto |
| `score` | [0 – 1000] pts | Muito Ruim, Ruim, Regular, Bom, Muito Bom |
| `tempo_emprego` | [0 – 30] anos | Curto, Moderado, Longo, Muito Longo |

### Variável de saída

| Variável | Universo | Termos linguísticos |
|----------|----------|---------------------|
| `risco` | [0 – 100] | Muito Baixo, Baixo, Médio, Alto, Muito Alto |

**Inferência:** Mamdani (min-implicação, max-agregação)  
**Defuzzificação:** Centroide  
**Regras:** 18 regras com base em conhecimento de domínio financeiro

---

## Estrutura do repositório

```
projeto/
├── app.py                          # App web interativo (Streamlit)
├── risco_credito_fuzzy_pso.ipynb   # Notebook de experimentos (PSO)
├── requirements.txt                # Dependências Python
├── resultados/
│   ├── fuzzy_antes/
│   │   ├── funcoes_pertinencia.png     # MFs do sistema padrão
│   │   └── superficie_controle.png    # Superfície de decisão 3D
│   ├── fuzzy_depois/
│   │   ├── mfs_otimizadas.png          # MFs antes vs depois do PSO
│   │   ├── impacto_mfs.png            # Análise de impacto do PSO nas MFs
│   │   └── params_otimizados.json     # Parâmetros otimizados (69 valores)
│   ├── evolucao/
│   │   ├── convergencia_pso.png       # Curva de convergência (run principal)
│   │   ├── convergencia_5runs.png     # 5 execuções independentes (PSO)
│   │   └── comparacao_pso_rs.png      # PSO vs Busca Aleatória
│   └── comparacao_cenarios.csv        # Tabela de resultados por cenário
└── docs/
    ├── relatorio_parte1_fuzzy.docx    # Relatório Parte 1
    └── relatorio_parte2_pso.docx      # Relatório Parte 2
```

---

## Instalação e execução

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o app interativo (recomendado)

```bash
streamlit run app.py
```

O app abre automaticamente no navegador em `http://localhost:8501`.

### 3. Executar o notebook (experimentos PSO)

```bash
jupyter notebook risco_credito_fuzzy_pso.ipynb
```

Execute **todas as células em ordem** para gerar os parâmetros otimizados e os gráficos.
O app carrega automaticamente os resultados do notebook (pasta `resultados/`).

Células de maior duração:

| Seção | Descrição | Tempo estimado |
|-------|-----------|----------------|
| Seção 9 (PSO principal) | 40 partículas, 150 iterações | ~12 s |
| Seção 14 (5 runs PSO) | 5 execuções independentes | ~60 s |
| Seção 15 (Busca Aleatória) | 5 execuções, 6 040 avaliações cada | ~150 s |

---

## Resumo dos resultados

### Comparação PSO vs Busca Aleatória (5 execuções, sementes [0, 7, 13, 21, 42])

| Método | MAE médio | Desvio | Melhor | vs Padrão |
|--------|-----------|--------|--------|-----------|
| Padrão (default) | 5,36 | — | 5,36 | baseline |
| **PSO** | ~0,51 | ~0,24 | ~0,20 | **−90,5%** |
| Busca Aleatória | > PSO | > PSO | > PSO | < PSO |

### Parâmetros do experimento

| Parâmetro | Valor |
|-----------|-------|
| Regras fuzzy | 18 |
| Cenários de teste | 10 |
| Dimensão do espaço PSO | 69 parâmetros (23 MFs × 3) |
| Partículas / Iterações | 40 / 150 |
| Inércia w / c1 / c2 | 0,7 / 1,5 / 1,5 |
| Execuções independentes | 5 |
| Sementes | [0, 7, 13, 21, 42] |
| Orçamento (avaliações) | 6 040 por execução |

---

## Tecnologias

- Python 3.12+
- [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) 0.5.0
- NumPy, Pandas, Matplotlib

---

## Declaração de uso de IA

O uso de IA generativa (Claude, Anthropic) foi empregado para auxiliar na estruturação do código, geração de comentários explicativos e revisão da lógica das regras fuzzy. Todo o código foi revisado, testado e validado pela equipe. A equipe assume integral responsabilidade pelo conteúdo final. Ver declaração completa no documento PDF.
