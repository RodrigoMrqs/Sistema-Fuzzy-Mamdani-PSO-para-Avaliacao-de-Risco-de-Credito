# Sistema Fuzzy Mamdani + PSO — Avaliação de Risco de Crédito

**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA 01/2026  
**AV2 Parte 1:** Sistema de Controle Fuzzy (Opção B — Produto)  
**AV2 Parte 2:** Computação Bioinspirada — PSO (Opção 2 — Protótipo, Alternativa 2: Integração Fuzzy-Evolutiva)

---

## Descrição

Sistema de apoio à decisão que avalia o **risco de crédito** de um solicitante de empréstimo usando lógica fuzzy Mamdani. Os parâmetros das funções de pertinência são otimizados automaticamente pelo algoritmo **PSO (Particle Swarm Optimization)**, que usa o próprio sistema fuzzy como função de aptidão.

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
├── risco_credito_fuzzy_pso.ipynb   # Notebook principal (execute este)
├── requirements.txt                # Dependências
├── resultados/
│   ├── fuzzy_antes/                # Gráficos do sistema original
│   ├── fuzzy_depois/               # Gráficos e parâmetros otimizados
│   └── evolucao/                   # Curvas de convergência do PSO
└── docs/                           # Relatórios
```

---

## Instalação e execução

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o notebook

Abra o Jupyter e execute o notebook principal:

```bash
jupyter notebook risco_credito_fuzzy_pso.ipynb
```

Ou via VS Code com a extensão Jupyter instalada.

### 3. Reproduzir todas as células em ordem

Execute as células de cima para baixo. As células de longa duração são:
- **Seção 9** — Execução do PSO (~10–15s com 40 partículas e 150 iterações)
- **Seção 13** — 5 execuções independentes (~50–60s total)

---

## Resumo dos resultados

| Métrica | Valor |
|---------|-------|
| Regras fuzzy | 18 |
| Cenários de teste | 10 |
| Dimensão do espaço PSO | 69 parâmetros |
| Partículas / Iterações | 40 / 150 |
| Execuções independentes | 5 |
| MAE antes (padrão) | 5.36 |
| MAE depois (PSO) | 0.51 (−90.5%) |
| Melhor run (5 execuções) | 0.20 (−96.3%) |

---

## Tecnologias

- Python 3.13
- [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/) 0.5.0
- NumPy, Pandas, Matplotlib

---

## Declaração de uso de IA

O uso de IA generativa (Claude) foi empregado para auxiliar na estruturação do código, geração de comentários e revisão da lógica das regras fuzzy. Todo o código foi revisado, testado e validado pela equipe. Ver seção de declaração no documento PDF.
