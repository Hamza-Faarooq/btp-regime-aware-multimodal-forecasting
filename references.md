# Day 0 Literature Check

This file records the literature reviewed for the BTP setup and gap check.

> Status: **provisional literature review**. The papers below have been searched and inspected from accessible abstracts/full-text pages. Before Step 0 is closed, the student should open the cited papers directly and confirm the notes against the papers themselves.

## 1. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models

**Araci (2019)** — [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)

- Focus: domain-specific pretrained language modelling for financial sentiment analysis.
- Main contribution: introduces FinBERT, a BERT-based model further adapted to financial text.
- Relevance: provides the text encoder foundation for M2/M3; supports using a financial-domain language model rather than generic sentiment features.
- Evaluation lesson: the paper is primarily about financial NLP performance, not about temporal evaluation robustness for market forecasting.

## 2. Decision support from financial disclosures with deep neural networks and transfer learning

**Kraus & Feuerriegel (2017)** — [Decision Support Systems, 104, 38–48](https://doi.org/10.1016/j.dss.2017.10.001) / [arXiv:1710.03954](https://arxiv.org/abs/1710.03954)

- Focus: predicting stock-price movements following financial disclosures with deep neural networks and transfer learning.
- Main contribution: compares traditional text representations with neural sequence models and studies transfer learning; LSTM performs strongly in their experiments.
- Relevance: establishes an important historical precedent for text-based financial movement prediction and transfer learning.
- Evaluation lesson: the authors explicitly note that comparisons across earlier studies can be difficult because studies use different filtering rules, train/test splits, metrics, and sometimes omit time-series cross-validation.
- This is particularly relevant to RQ1 because it documents comparability problems in the existing literature.

## 3. FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis

**Halder (2022)** — [arXiv:2211.07392](https://arxiv.org/abs/2211.07392)

- Focus: combining financial/news sentiment with LSTM-based stock prediction.
- Data described: NASDAQ-100 index data and New York Times news articles.
- Models: MLP, LSTM, and FinBERT-LSTM.
- Metrics: MAE, MAPE, and accuracy.
- Relevance: close precedent for M2/M3-style price + financial-text modelling.
- Evaluation lesson: the paper's headline contribution is model integration/predictive performance rather than a systematic comparison of valid versus invalid temporal evaluation protocols.

## 4. Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis

**Gu et al. (2024)** — [arXiv:2407.16150](https://arxiv.org/abs/2407.16150)

- Focus: combining historical stock information with financial news categories and FinBERT sentiment.
- Data described: NASDAQ-100 stock data and Benzinga news.
- News is separated into market-, industry-, and stock-related categories.
- Relevance: another direct precedent for adding financial news to historical price information.
- Evaluation lesson: again, the main emphasis is predictive improvement from news/model design rather than measuring how conclusions change under multiple temporal evaluation protocols.

## 5. Multimodal Stock Price Prediction

**Karadaş, Eravcı & Özbayoğlu (2025)** — [arXiv:2502.05186](https://arxiv.org/abs/2502.05186)

- Focus: multimodal stock prediction using traditional financial metrics, tweets, and news.
- Text sentiment is obtained using ChatGPT-4o and FinBERT.
- The multimodal streams are used to augment an LSTM baseline.
- Relevance: directly supports the thesis framing of comparing price-only and price+news systems.
- Reported result: the authors report improvements of up to 5% from incorporating the additional modalities.
- Evaluation lesson: the paper demonstrates multimodal predictive gains, but does not make evaluation-protocol sensitivity the central experimental question.

## 6. Stock Movement Prediction with Multimodal Stable Fusion via Gated Cross-Attention Mechanism

**Zong & Zhou (2024)** — [arXiv:2406.06594](https://arxiv.org/abs/2406.06594)

- Focus: multimodal stock movement prediction using indicator sequences, dynamic documents, and relational information.
- Architecture: trimodal encoding plus gated cross-attention fusion.
- Relevance: demonstrates the current trend toward increasingly sophisticated multimodal fusion architectures.
- Evaluation lesson: useful contrast for this BTP because the BTP deliberately does **not** claim architectural novelty; it asks whether conclusions are robust to evaluation choices and whether news adds value across regimes.

## 7. Multimodal information fusion for financial forecasting via cross-attention and calibrated uncertainty

**Bustarviejo & Bousoño-Calzón (2026)** — *Machine Learning with Applications*, 23, 100840. [Publisher page](https://www.sciencedirect.com/science/article/pii/S2666827026000058)

- Focus: multimodal financial forecasting using market dynamics, company-level indicators, and daily news embeddings.
- Architecture: frozen Chronos-T5 market encoder plus projection layers and bidirectional cross-attention.
- Evaluation: includes experiments across currency pairs and market regimes.
- Relevance: especially important because it explicitly recognizes regime dependence in multimodal forecasting.
- Evaluation lesson: it strengthens the argument that regime-aware analysis is timely, but its primary contribution is a new multimodal probabilistic forecasting framework rather than a controlled audit of evaluation protocols.

## 8. FusionLSTM-CNF: a confidence-calibrated multi-modal late fusion framework for robust stock movement prediction under uncertainty

**Wang et al. (2026)** — *Scientific Reports*. [Article](https://www.nature.com/articles/s41598-026-43381-3)

- Focus: multimodal stock movement prediction with technical indicators, FinBERT-derived news sentiment, and cross-asset correlations.
- Evaluation spans S&P 500, NASDAQ, and FTSE 100 and explicitly considers bull, bear, recovery, and stable-growth regimes.
- Relevance: strong recent precedent for regime-aware multimodal financial prediction.
- Evaluation lesson: the study reports multiple independent runs and extensive metrics, but its central contribution remains the proposed fusion/uncertainty architecture. It therefore provides a useful comparison point for a thesis whose novelty is the evaluation audit rather than another fusion architecture.

## 9. The Probability of Backtest Overfitting

**Bailey, Borwein, López de Prado & Zhu (2015)** — *Journal of Computational Finance*. [UC eScholarship](https://escholarship.org/uc/item/4w1110bb) / [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)

- Focus: probability of backtest overfitting and the unreliability of selecting strategies using historical performance.
- Main contribution: introduces combinatorially symmetric cross-validation (CSCV) as a framework for estimating probability of backtest overfitting.
- Relevance: provides methodological motivation for treating evaluation design and repeated experimentation as sources of performance inflation.
- Evaluation lesson: an apparently strong historical result is not automatically evidence of genuine out-of-sample predictive ability.

## 10. Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance

**Bailey, Borwein, López de Prado & Zhu (2014)** — *Notices of the American Mathematical Society*, 61(5), 458–471. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659)

- Focus: how trying many strategy configurations can produce impressive but misleading in-sample/backtest performance.
- Main contribution: formalizes and illustrates the relationship between the number of configurations tried and the likelihood of backtest overfitting.
- Relevance: supports the BTP's decision to freeze the experimental design and avoid test-set tuning.
- Evaluation lesson: reporting only the best result without transparent control of the evaluation process can inflate perceived performance.

---

## Cross-paper synthesis

### What the literature clearly establishes

1. **Financial text can contain predictive information.**  
   Multiple studies use financial news/disclosures and neural language models for stock movement or price prediction.

2. **FinBERT is a defensible text representation for this BTP.**  
   It is specifically adapted to financial language and is widely used in subsequent financial-text forecasting work.

3. **Multimodal financial forecasting is an active research area.**  
   Recent work combines market histories with news, sentiment, technical indicators, social data, relational information, or other modalities.

4. **Regime dependence is increasingly recognized.**  
   Recent multimodal forecasting papers explicitly evaluate or discuss different market regimes.

5. **Evaluation design matters in financial ML.**  
   The financial ML/backtesting literature highlights overfitting, selection bias, and the dangers of treating historical evaluation as automatically reliable.

6. **Comparisons across financial prediction papers are often difficult.**  
   Kraus & Feuerriegel explicitly note differences in filtering, train/test splits, metrics, and time-series validation as reasons why results from prior studies may not be directly comparable.

### What this literature does NOT yet establish

The papers reviewed here do **not** establish that the same multimodal market-prediction pipeline has been systematically subjected to a controlled comparison of:

- random splitting with all-data preprocessing,
- chronological splitting with all-data preprocessing, and
- chronological splitting with train-only preprocessing,

while also measuring the incremental contribution of news separately across market regimes.

That is the specific empirical space occupied by this BTP.

## Provisional gap sentence

> **Prior work demonstrates the usefulness of financial news and multimodal models for market prediction, while financial-ML research separately documents the risks of backtest overfitting and inconsistent evaluation; fewer studies directly quantify how evaluation protocol changes multimodal market-prediction results within the same controlled pipeline and then test whether news adds value consistently across market regimes.**

## Step 0 reading checklist

Before declaring Step 0 complete, open the papers above and verify the notes against the original sources. In particular, confirm:

- [ ] FinBERT (2019)
- [ ] Kraus & Feuerriegel (2017)
- [ ] Halder (2022)
- [ ] Gu et al. (2024)
- [ ] Multimodal Stock Price Prediction (2025)
- [ ] Zong & Zhou (2024)
- [ ] Bustarviejo & Bousoño-Calzón (2026)
- [ ] FusionLSTM-CNF (2026)
- [ ] Probability of Backtest Overfitting (2015)
- [ ] Pseudo-Mathematics and Financial Charlatanism (2014)

**Step 0 status:** Environment/repository setup is complete. Literature review and gap identification are provisionally complete; final Step 0 closure requires the reading/checklist confirmation above. Do not begin Step 1 until this checkpoint is closed.
