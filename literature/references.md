# Literature: candidate reading list and research-gap notes

> **Status: reading list, not a completed literature review.**
> Bibliographic details below were checked against arXiv / publisher / SSRN / PMC records on **21 Oct 2026**
> (see the "Verification" column). The one-line descriptions come **only from abstracts and landing pages**.
> **None of these papers has been read in full by the author.** Before any paper is cited in the thesis it must be
> opened and read (at minimum abstract, introduction, method overview and conclusion); tick the box in
> [`papers.csv`](papers.csv) (`opened_by_me`) once that is true and rewrite the notes in your own words.

## Verification summary

| # | Paper | Verification (21 Oct 2026) |
|---|---|---|
| 0 | Saqur et al. (2024), *NIFTY Financial News Headlines Dataset*, [arXiv:2405.09747](https://arxiv.org/abs/2405.09747) | Title, authors and dataset card confirmed via arXiv/Hugging Face listings. **This is the dataset paper and must be read.** |
| 1 | Araci (2019), *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*, [arXiv:1908.10063](https://arxiv.org/abs/1908.10063) | Confirmed (title, author, Aug 2019). |
| 2 | Kraus & Feuerriegel (2017), *Decision support from financial disclosures with deep neural networks and transfer learning*, [arXiv:1710.03954](https://arxiv.org/abs/1710.03954) | Confirmed on arXiv (title, authors, Oct 2017). Journal details (Decision Support Systems 104, 38-48) not re-checked. |
| 3 | Halder (2022), *FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis*, [arXiv:2211.07392](https://arxiv.org/abs/2211.07392) | Confirmed (NASDAQ-100 + New York Times; MLP, LSTM, FinBERT-LSTM; MAE, MAPE, accuracy). |
| 4 | Gu et al. (2024), *Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis*, [arXiv:2407.16150](https://arxiv.org/abs/2407.16150) | Confirmed (first author Wenjun Gu; NASDAQ-100 + Benzinga). |
| 5 | Karadas, Eravci & Ozbayoglu (2025), *Multimodal Stock Price Prediction*, [arXiv:2502.05186](https://arxiv.org/abs/2502.05186) | Confirmed (tweets + news scored with ChatGPT-4o and FinBERT; abstract reports up to 5% improvement over the LSTM baseline). |
| 6 | Zong et al. (2024), *Stock Movement Prediction with Multimodal Stable Fusion via Gated Cross-Attention Mechanism*, [arXiv:2406.06594](https://arxiv.org/abs/2406.06594) | Title confirmed. **Author list differs between arXiv versions** (v1 PDF: Zong, Shao, Lu, Zhuang; v2 HTML: Zong and Zhou). Cite the author list printed on the version you actually read. |
| 7 | Bustarviejo & Bousono-Calzon (2026), *Multimodal information fusion for financial forecasting via cross-attention and calibrated uncertainty*, *Machine Learning with Applications* | Title and ScienceDirect record found by search; **publisher page could not be opened** from the tool used, so authors, volume and article number are **not verified**. Check before citing. |
| 8 | Wang et al. (2026), *FusionLSTM-CNF: a confidence-calibrated multi-modal late fusion framework for robust stock movement prediction under uncertainty*, *Scientific Reports* 16, 17494, [DOI 10.1038/s41598-026-43381-3](https://doi.org/10.1038/s41598-026-43381-3) | Confirmed on PMC (published 9 Apr 2026; S&P 500, NASDAQ, FTSE 100; four regimes named in the main text; FinBERT news embeddings; the paper is internally inconsistent about the number of runs, 5 vs 10). |
| 9 | Bailey, Borwein, Lopez de Prado & Zhu (2015), *The Probability of Backtest Overfitting*, *Journal of Computational Finance*, [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) | Confirmed (CSCV framework; journal listed as forthcoming in 2015 on SSRN). |
| 10 | Bailey, Borwein, Lopez de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*, *Notices of the AMS* 61(5), 458-471, [SSRN 2308659](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659) | Confirmed. |

## What each paper is used for (from abstracts; to be replaced by your own notes after reading)

- **Dataset (0):** source and construction of the benchmark used here (SPY market context + curated daily headlines, Rise/Neutral/Fall labels).
- **Financial text representation (1):** FinBERT is a BERT model adapted to financial text; it motivates using a frozen financial-domain encoder.
- **Text-based movement prediction (2):** earlier deep-learning work on stock reactions to financial text; the authors note that comparing studies is difficult because of differing filters, splits and metrics, which is the motivation for RQ1.
- **Price + news precedents (3, 4, 5):** FinBERT/sentiment features combined with LSTM price models; headline contribution is predictive gain from news, not sensitivity to evaluation protocol.
- **More elaborate fusion (6, 7, 8):** gated cross-attention, cross-attention with calibrated uncertainty, and confidence-calibrated late fusion. They show the field's direction toward complex fusion; this thesis deliberately keeps the architecture small. (8) evaluates several market regimes, (7) reportedly also reports regime experiments (unverified).
- **Evaluation and overfitting (9, 10):** why a single historical backtest result is not automatically evidence of out-of-sample ability, and why the number of configurations tried should be disclosed. This motivates freezing hyper-parameters and reporting every model that was run.

## Cautious synthesis (re-check after reading)

1. Financial text can carry predictive information, and FinBERT-style encoders are a common choice.
2. Multimodal price + news forecasting is an active area; several recent papers use regimes or adaptive fusion.
3. Evaluation design matters in financial ML, and cross-paper comparisons are often hard.

### What these abstracts do not obviously show

Within the papers listed, none was found whose main aim is a **controlled comparison of random vs chronological splitting and all-data vs train-only preprocessing inside one multimodal pipeline, combined with a regime-wise analysis of the incremental value of news.** This is *not* a claim that no such study exists: only that this reading list does not contain one. A broader search (e.g. on data leakage / look-ahead bias in financial machine learning) is still needed and any closer work found must be cited and the gap sentence adjusted.

### Provisional gap sentence (edit after reading)

> Prior work shows that financial news and multimodal models can help market prediction, while financial-ML research separately documents backtest overfitting and inconsistent evaluation. This thesis examines, within one controlled pipeline on a public benchmark, how much reported performance changes with the evaluation protocol, and whether the incremental value of news differs across volatility regimes, including the February to April 2020 stress period.

## Still to do

- Open and read every paper in [`papers.csv`](papers.csv); fill the notes in your own words.
- Run the additional search on data leakage / look-ahead bias in stock prediction and add what you find.
- Do not describe this work as "the first" to do anything.
