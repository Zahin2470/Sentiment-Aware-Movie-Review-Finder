<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1B2A,50:0891B2,100:14B8A6&height=220&section=header&text=SentiVec&fontSize=72&fontColor=FFFFFF&fontAlignY=38&desc=Sentiment-Aware%20Vector-Based%20Retrieval%20System&descSize=18&descAlignY=62&animation=fadeIn" width="100%"/>
</div>

<p align="center">
<img src="https://img.shields.io/badge/Version-3.0%20Reverse%20Edition-0891B2?style=for-the-badge&logo=bookstack&logoColor=white"/>
<img src="https://img.shields.io/badge/Directions-B%20%2B%20D%20(Amazon%20Train)-14B8A6?style=for-the-badge&logo=read-the-docs&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FAISS-ANN%20Search-FF6F00?style=for-the-badge&logo=meta&logoColor=white"/><br/>
<img src="https://img.shields.io/badge/Train-Amazon%2050K%20Reviews-10b981?style=for-the-badge&logo=amazon&logoColor=white"/>
<img src="https://img.shields.io/badge/Test%20B-IMDB%20Queries-3b82f6?style=for-the-badge&logo=imdb&logoColor=white"/>
<img src="https://img.shields.io/badge/Test%20D-Amazon%20Queries-f97316?style=for-the-badge&logo=amazon&logoColor=white"/>
<img src="https://img.shields.io/badge/4--Way%20Matrix-Complete-8B5CF6?style=for-the-badge&logo=databricks&logoColor=white"/>
</p>

---

## 📌 About This Notebook

This notebook is the **reverse companion** to `SentiVec_Q2_Journal_Complete.ipynb`. Together they constitute the **complete four-way cross-dataset validation** required for Q2 journal submission.

| Notebook | Directions | Train Corpus | Test Corpus |
|----------|-----------|-------------|-------------|
| **Forward** (`Sentiment Aware Vector IMDB-Amazon.ipynb`) | **A** + **C** | IMDB 44,620 | IMDB / Amazon |
| **Reverse** (`Sentiment Aware Vector Amazon-IMDB.ipynb`) | **B** + **D** | Amazon 50,000 | IMDB / Amazon |

> 💡 **Core Question**: Is the SentiVec sentiment-filtering benefit consistent regardless of which corpus forms the retrieval index? The four-way matrix answers this definitively.

---

## 🎯 Four-Way Cross-Dataset Matrix

```
                        ┌─────────────────────────────────────┐
                        │          TEST DOMAIN                │
                        ├──────────────────┬──────────────────┤
                        │   IMDB Test      │   Amazon Test    │
          ┌─────────────┼──────────────────┼──────────────────┤
  TRAIN   │ IMDB Train  │ A — in-domain    │ C — cross test   │
  DOMAIN  │             │ (Forward NB) ✓   │ (Forward NB) ✓   │
          ├─────────────┼──────────────────┼──────────────────┤
          │ Amazon Train│ B — cross train  │ D — in-domain    │
          │             │ (THIS NB) ✓      │ (THIS NB) ✓      │
          └─────────────┴──────────────────┴──────────────────┘
```

---

## 👥 Authors & Affiliation

<table width="100%">
<tr>
<td width="33%" align="center">
<br/>
<img src="https://img.shields.io/badge/%F0%9F%A7%91%E2%80%8D%F0%9F%92%BB%20Lead%20Author-Abrar%20Hossain%20Zahin-0D1B2A?style=for-the-badge"/>
<br/><br/>
<b>Abrar Hossain Zahin</b><br/>
<em>System Design · Experiments · Writing</em>
<em>Analysis · Statistical</em><br/><br/>
<a href="https://abrar-hossain-zahin-portfolio.vercel.app/">
<img src="https://img.shields.io/badge/Portfolio-0891B2?style=flat-square&logo=vercel&logoColor=white"/>
</a>
<a href="https://www.kaggle.com/mdabrarhossainzahin">
<img src="https://img.shields.io/badge/Kaggle-0891B2?style=flat-square&logo=vercel&logoColor=white"/>
</a>
<br/><br/>
<img src="https://img.shields.io/badge/East%20West%20University-CSE-0891B2?style=flat-square"/>
</td>
<td width="33%" align="center">
<br/>
<img src="https://img.shields.io/badge/%F0%9F%A7%91%E2%80%8D%F0%9F%92%BB%20Co--Author-KM%20Fahim%20A.%20Bari-0D1B2A?style=for-the-badge"/>
<br/><br/>
<b>KM Fahim A. Bari</b><br/>
<em>Evaluation · Visualization</em><br/><br/>
<br/>
<img src="https://img.shields.io/badge/East%20West%20University-CSE-0891B2?style=flat-square"/>
</td>
<td width="33%" align="center">
<br/>
<img src="https://img.shields.io/badge/%F0%9F%A7%91%E2%80%8D%F0%9F%92%BB%20Co--Author-Mohammad%20Rezwanul%20Huq-0D1B2A?style=for-the-badge"/>
<br/><br/>
<b>Mohammad Rezwanul Huq</b><br/>
<em>Correspnder Validation</em><br/><br/>
<br/>
<img src="https://img.shields.io/badge/East%20West%20University-CSE-0891B2?style=flat-square"/>
</td>
</tr>
</table>

<div align="center">
<img src="https://img.shields.io/badge/%F0%9F%9B%8F%EF%B8%8F%20East%20West%20University-Dept.%20of%20Computer%20Science%20%26%20Engineering-0D1B2A?style=for-the-badge"/> &nbsp;
<img src="https://img.shields.io/badge/%F0%9F%93%85%20June%202026-Version%203.0-14B8A6?style=for-the-badge"/>
</div>

---

## ✅ Q2-Level Research Standards (This Notebook)

<table width="100%">
<tr>
<td width="25%" valign="top">
<b>🔬 Experimental Rigor</b><br/><br/>
<ul>
<li>All test queries)</li>
<li>Paired t-tests + Wilcoxon</li>
<li>Cohen's d effect sizes</li>
<li>95% confidence intervals</li>
<li>Fixed seed = 42</li>
<li>Directions B + D covered</li>
</ul>
</td>
<td width="25%" valign="top">
<b>📊 Evaluation Coverage</b><br/><br/>
<ul>
<li><b>4</b> FAISS index types</li>
<li><b>3</b> sentiment classifiers</li>
<li><b>8</b> threshold values</li>
<li><b>5</b> ablation configs</li>
<li><b>12</b> adversarial queries</li>
<li><b>5</b> baseline methods</li>
</ul>
</td>
<td width="25%" valign="top">
<b>🧠 Advanced Analysis</b><br/><br/>
<ul>
<li>Failure taxonomy</li>
<li>Latency profiling</li>
<li>Memory footprint</li>
<li>Domain shift analysis</li>
<li>Bidirectional comparison</li>
<li>Four-way matrix figure</li>
</ul>
</td>
<td width="25%" valign="top">
<b>🔁 Reproducibility</b><br/><br/>
<ul>
<li>Fixed random seed (42)</li>
<li>Kaggle-compatible env</li>
<li>Auto-recovery guards</li>
<li>CSV-based data loading</li>
<li>LaTeX tables exported</li>
</ul>
</td>
</tr>
</table>

---

## 🏗️ Pipeline: Reverse Direction

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 SENTIVEC REVERSE PIPELINE v3.0                           │
├─────────────────┬────────────────────────────┬───────────────────────────┤
│   TRAIN CORPUS  │      RETRIEVAL INDEX       │    TEST QUERIES           │
│                 │                            │                           │
│  Amazon Reviews │  Sentence-BERT             │  Direction B: IMDB 4,958  │
│  50,000 reviews │  (all-MiniLM-L6-v2)        │  Direction D: Amazon 5,000│
│  (balanced)     │  → 384-dim embeddings      │                           │
│       │         │  → L2 normalised           │  Query encoding (same     │
│       ▼         │                            │  model, no retraining)    │
│  FAISS Indexes  │  ┌──────────────────────┐  │           │               │
│  on Amazon      │  │ FlatL2  │ exact      │  │           ▼               │
│  corpus         │  │ IVF     │ approx     │  │  Semantic + Sentiment     │
│                 │  │ HNSW    │ graph ANN  │  │  Retrieval Pipelines      │
│                 │  │ IVFPQ   │ compressed │  │           │               │
│                 │  └──────────────────────┘  │           ▼               │
│                 │                            │  Evaluation + Stats       │
└─────────────────┴────────────────────────────┴───────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │         FOUR-WAY COMPARISON             │
              │  A (Fwd NB) + B + C (Fwd NB) + D        │
              │  Complete 2×2 cross-dataset matrix      │
              └─────────────────────────────────────────┘
```

---

## 📊 Notebook Cell Map

| Cell | Section | Content |
|------|---------|---------|
| 1 | Setup | Environment, packages, GPU detection |
| 2 | Amazon Corpus | Load 50K from `train.ft.txt.bz2` (FastText format) |
| 3 | IMDB Queries | Load test set from CSV (Direction B queries) |
| 4 | Embeddings | all-MiniLM-L6-v2 on both corpora |
| 5 | FAISS Indexes | FlatL2 / IVF / HNSW / IVFPQ on Amazon |
| 6 | Classifiers | DistilBERT + VADER + TextBlob |
| 7 | Retrieval | Without/with sentiment functions |
| 8 | Evaluation | Metrics + ground truth + auto-recovery |
| 9 | **Direction B** | Amazon train → IMDB test |
| 10 | Statistical Tests | t-test, Wilcoxon, Cohen's d, 95% CI |
| 11 | Ablation Study | 5 configurations |
| 12 | Threshold Analysis | 8 thresholds (0.50–0.90) |
| 13 | Classifier Comparison | DistilBERT vs VADER vs TextBlob |
| 14 | Mixed-Tone + Errors | 12 adversarial queries + failure taxonomy |
| 15 | Baselines | SparseBM25, TF-IDF, Hybrid |
| 16 | Latency + Memory | Per-component profiling |
| 17 | Results Summary | LaTeX table export |
| 18 | **Bidirectional (A vs B)** | Reads A from forward CSV|
| 19 | Three-Way Figure | A + B + C combined |
| 20 | Execution Summary | Output inventory |
| 21 | Amazon Test Load | Load test.ft.txt.bz2 (Direction D) |
| 22 | Amazon Test Emb | Generate embeddings for Amazon test |
| 23 | **Direction D** | Amazon train → Amazon test |
| 24 | Direction D Stats | t-test, Wilcoxon, Cohen's d |
| 25 | **Four-Way (A+B+C+D)** | Reads A+C from forward CSV|
| 26 | Final Summary | 2×2 matrix + paper paragraph template |

---

## ⏱️ Estimated Runtime

| Mode | Time |
|------|------|
| Full run (T4 GPU, Kaggle) | ~90–120 minutes |
| CPU only | ~3–4 hours |

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14B8A6,50:0891B2,100:0D1B2A&height=100&section=footer&fontSize=13&fontColor=FFFFFF&text=SentiVec%20v3.0%20Reverse%20%7C%20East%20West%20University%20%7C%20June%202026&fontAlignY=65&animation=fadeIn" width="100%"/>
</div>"""
