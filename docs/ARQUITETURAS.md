# Arquiteturas e hiperparâmetros

Documento de referência para a seção de métodos. Todos os valores abaixo foram extraídos
do código-fonte do projeto (`src/models/`, `notebooks/train_all.py`, `src/utils/splits.py`)
e, quando o projeto não sobrescreve um parâmetro, dos defaults da classe correspondente do
`pytorch-forecasting 1.8.0` — versão usada nas runs. Nada aqui é estimado.

---

## 1. Protocolo experimental

| Item | Valor |
|---|---|
| Granularidade | mensal |
| Período | 2000-01 a 2023-12 (288 meses) |
| Unidades espaciais | 23 municípios de SP |
| Agravos | 5 (hanseníase, hepatite, HIV/AIDS, sífilis, tuberculose) |
| Alvo | contagem mensal de casos (`n_<doenca>`) |
| Horizontes de previsão | h ∈ {1, 3, 6, 12} meses |
| Validação | rolling origin, 4 origens: 2019-12, 2020-12, 2021-12, 2022-12 |
| Janela de teste | `[T+1, T+H]` a cada origem T (H = 12) |
| Janela de validação (early stopping) | `[T−H+1, T]` — separada do teste |

## 2. Janelas de entrada e saída (deep models)

| Parâmetro | Valor | Observação |
|---|---|---|
| `max_encoder_length` | **36 meses** | igual para todos os 5 deep models |
| `min_encoder_length` | **18** (TFT, DeepAR, LSTM, GRU) / **36** (N-HiTS) | N-HiTS usa janela fixa (`fixed_lengths=True`) |
| `max_prediction_length` | **12 meses** | = max(horizontes) |
| `min_prediction_length` | **1** (TFT, DeepAR, LSTM, GRU) / **12** (N-HiTS) | idem |

## 3. Treinamento (comum aos 5 deep models)

| Parâmetro | Valor efetivo |
|---|---|
| Batch size | **128** |
| Learning rate inicial | **1e-3** (mesmo para os 5 modelos) |
| Épocas (teto) | **200** |
| EarlyStopping | monitor `val_loss`, **patience = 7**, `min_delta = 1e-5` |
| Scheduler | `ReduceLROnPlateau`, patience = 5 |
| Gradient clipping | 0.1 |
| Precisão | 32 bits |
| Semente | 42 (`seed_everything`, `deterministic="warn"`) |

> `max_epochs=200` é apenas o teto: o EarlyStopping interrompe bem antes na prática. A época
> exata de parada por run está nos logs do Lightning (`lightning_logs/`), não nos CSVs de
> resultado.

> `deterministic="warn"` (e não `True`) porque o backward do N-HiTS não tem kernel
> determinístico em GPU e abortaria o treino. Com semente fixa a reprodutibilidade continua
> alta, mas não é bit-a-bit garantida entre execuções.

## 4. Topologia por modelo

### 4.1 Explicitado no código do projeto

| Modelo | Configuração | Função de perda |
|---|---|---|
| **TFT** | `hidden_size=32`, `attention_head_size=4`, `hidden_continuous_size=16`, `dropout=0.2` | `QuantileLoss` (7 quantis; mediana usada como previsão pontual) |
| **N-HiTS** | `hidden_size=64` (dobro do padrão do projeto), `dropout=0.2` | `QuantileLoss` |
| **DeepAR** | `hidden_size=32`, `rnn_layers=2`, `dropout=0.2` | `NegativeBinomialDistributionLoss` |
| **LSTM** | `cell_type="LSTM"`, `hidden_size=32`, `rnn_layers=2`, `dropout=0.2` | `MAE` |
| **GRU** | `cell_type="GRU"`, `hidden_size=32`, `rnn_layers=2`, `dropout=0.2` | `MAE` |

### 4.2 Herdado dos defaults do `pytorch-forecasting 1.8.0`

O projeto não fixa estes parâmetros; eles vêm da biblioteca.

**N-HiTS**
- `n_blocks = [1, 1, 1]` → 3 stacks, 1 bloco por stack
- `n_layers = 2` → camadas MLP por bloco
- `pooling_sizes = [8, 4, 1]` e `downsample_frequencies = [12, 8, 1]` — **calculados em tempo
  de execução em função do horizonte**; os valores citados correspondem a H=12 (o do projeto).
  Se o horizonte máximo mudar, esses números mudam automaticamente.
- ativação ReLU, inicialização `lecun_normal`, `naive_level=True`, `shared_weights=True`,
  `batch_normalization=False`

**TFT**
- `lstm_layers = 1` — uma camada de LSTM no encoder e no decoder
- `causal_attention = True` — a atenção não acessa passos futuros dentro da janela de decodificação
- `share_single_variable_networks = False` — cada variável tem sua própria rede de seleção

**DeepAR**
- `cell_type = LSTM` (default) — o DeepAR aqui é um LSTM autorregressivo com cabeça
  probabilística Negative Binomial, não uma célula RNN simples

## 5. Covariáveis por modelo

| Modelo | Alvo | Calendário (`time_idx`, `month_of_year`, `covid_period`) | Clima (encoder) | Socioeconômico (estático) |
|---|---|---|---|---|
| TFT | ✅ | ✅ | ✅ | ✅ |
| N-HiTS | ✅ | ✅ | ✅ | ✅ |
| DeepAR | ✅ | ✅ | ❌ | ❌ |
| LSTM / GRU | ✅ | ✅ | ❌ | ❌ |

Clima entra apenas como variável **desconhecida no futuro** (só encoder), nunca como
covariável futura conhecida — ver a seção de vazamentos corrigidos no `README.md`.

DeepAR, LSTM e GRU são autorregressivos e exigem que encoder e decoder tenham as mesmas
variáveis além do alvo, o que impede covariável desconhecida — por isso rodam com alvo +
calendário apenas.

## 6. Modelos de árvore (LightGBM, CatBoost, XGBoost)

Esquema **direto as-of-origem**: para a origem T e horizonte h, as features são da linha T
(valor em T, lags ≤ T, clima em T e defasado ≤ T) e o alvo é o valor em T+h. Um modelo por
horizonte.

| Modelo | Configuração | Objetivo |
|---|---|---|
| **LightGBM** | `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `min_data_in_leaf=20`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5` | `poisson` |
| **CatBoost** | `iterations=600`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0` | `Poisson` |
| **XGBoost** | `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, `tree_method=hist` | `count:poisson` |

Features: lags do alvo (0, 1, 2, 3, 6, 12, 13), médias móveis (3, 6, 12), clima
contemporâneo e defasado (0, 1, 3, 12), socioeconômicas, mês-alvo e identificador do
município (ambos categóricos).

## 7. Modelos estatísticos

| Modelo | Configuração |
|---|---|
| **SeasonalNaive** | `season_length=12` |
| **SeasonalWindowAverage (MA3)** | `season_length=12`, `window_size=3` |
| **AutoARIMA / SARIMA** | `statsforecast`, `season_length=12` — seleção automática de ordem; cobre ARIMA como caso particular |
| **Prophet** | univariado por município, sazonalidade anual aditiva, sem sazonalidade semanal/diária; previsões truncadas em zero |

## 8. Tuning de hiperparâmetros

**Não houve busca sistemática de hiperparâmetros.** Todos os valores acima foram fixados
manualmente e mantidos iguais entre modelos, doenças e horizontes (com as poucas exceções
já indicadas, como o `hidden_size` dobrado do N-HiTS). Não há uso de Optuna, grid search ou
random search em nenhum ponto do código — busca automatizada consta apenas como item de
trabalho futuro no roadmap do `README.md`.

Para o texto do artigo, a formulação correta é: *"os hiperparâmetros foram definidos
manualmente, sem busca sistemática; a otimização automatizada permanece como trabalho
futuro."*
