# KGIRS Viva Quick Revision Cheatsheet (The Complete 6-Module Story + Lab)

> **Course:** `26CMPE752` — Knowledge Graph and Information Retrieval Systems  
> **Read in Order:** [Module 1](Module_1_Introduction_to_IR_and_Graphs.md) → [Module 2](Module_2_IR_Modeling_Indexing_and_Ranking.md) → [Module 3](Module_3_Cross_Modal_KG_Fundamentals.md) → [Module 4](Module_4_Multimodal_KG_Construction.md) → [Module 5](Module_5_Advanced_Retrieval_Architectures.md) → [Module 6](Module_6_GraphRAG_Evaluation_and_Explainability.md)

---

## 1. The Whole Subject as One Simple Story (Read This Before Walking Into Your Viva!)

1. **[Module 1 — Finding Needles in Messy Haystacks](Module_1_Introduction_to_IR_and_Graphs.md):**
   We start with 50,000 messy files. Raw numbers (`92`) become **Information** (`Rahul: 92%`) and **Knowledge** (`Not eligible for exam`). Scanning files one-by-one is slow, so we build a back-of-the-book **Inverted Index**. Users search via **Keywords**, **Patterns** (Regex/Fuzzy Levenshtein), or **Structured Queries** (SQL/Cypher/SPARQL). Because keywords miss synonyms (`car` vs `automobile`), we map text, photos, and audio onto a shared **Meaning Map** (**Dense & Multimodal Embeddings / CLIP**), keep **BM25** for exact codes (`E4021`), keep per-word vectors for accuracy (**ColBERT Late Interaction**), and use **GQA**, **Longformer Sliding Window**, and **FlashAttention** so AI can read huge documents without running out of GPU memory.

2. **[Module 2 — Scoring, Scaling to Billions & Running at Companies](Module_2_IR_Modeling_Indexing_and_Ranking.md):**
   Once search finds 500 matching files, how do we rank #1 vs #2? We try arrow angles (**Vector Space Model + TF-IDF**), odds (**Probabilistic Model / BIM**), and the keyword king (**BM25** with saturation $k_1$ and length fairness $b$). To bridge meaning gaps, we use **Bi-Encoders** (fast shortlist) + **Cross-Encoders** (smart careful re-ranker) in a **Two-Stage Funnel**. At 100M+ vectors, brute-force scanning breaks, so we organize vectors with **IVF** (clusters), **HNSW** (highway-to-street graph), and **PQ** (tiny 1-byte zip codes). We mix BM25 + Dense via **RRF**, respect permissions with **Metadata Filtering**, align the LLM cheaply with **LoRA / SFT / DPO**, and pick real company engines (**Elasticsearch, OpenSearch, Postgres `pgvector`, Qdrant, LanceDB**).

3. **[Module 3 — Storing Exact Connected Facts (Knowledge Graphs)](Module_3_Cross_Modal_KG_Fundamentals.md):**
   Paragraph search is vague for multi-hop questions (`"Which drugs treat diseases sharing cough & breathlessness?"`). So we store exact 3-word facts—**Triples `(Subject, Predicate, Object)`**—connected as **Nodes, Edges & Properties**, including image and audio nodes (**Multimodal KG**). We compare developer-friendly **LPG (Neo4j/Cypher)** vs. open-web **RDF (URIs/SPARQL)**, connect domains (**Cross-domain** & **Cross-modal**), follow **Schemas / Ontologies** while dodging **Anti-patterns** (missing domain/range, double-meaning classes, verb shortcuts, giant supernodes), let **OWL reasoners (HermiT/Pellet)** infer new facts under the **Open World Assumption**, and grade generated text with **Perplexity, BLEU, ROUGE, METEOR, BERTScore & LLM-as-Judge**.

4. **[Module 4 — Running a Graph DB & Filling It Automatically From Messy PDFs](Module_4_Multimodal_KG_Construction.md):**
   Now we build it! Native Graph DBs (Neo4j) crush SQL on 4–5 hop queries because of **Index-Free Adjacency** ($O(1)$ direct pointers per hop instead of `JOIN` indexes). We query with **Cypher (`MERGE`)**, explore with **BFS** (shortest hops) & **DFS**, pull entities/relations from text (NER, RE, Entity Linking), merge duplicates like `"A.P.J. Kalam"` using **Blocking + Token Jaccard / Levenshtein / Embeddings (Entity Resolution)**, force LLMs to obey strict rules with **Pydantic + Instructor retries**, and read scanned PDFs/tables/stamps using **OCR + LayoutLM + Table Transformer + VLMs**.

5. **[Module 5 — Fixing the 7 Ways Basic RAG Breaks](Module_5_Advanced_Retrieval_Architectures.md):**
   Basic RAG breaks in real life! We upgrade it step-by-step: stop chopping answers in half (**Fixed → Recursive → Semantic Chunking**), fix vague questions and buried noise (**HyDE, Query Rewriting, Reranking**), stamp missing document identity onto chunks (**Anthropic Contextual Retrieval**), add a quality bouncer + web search fallback (**CRAG**), let the LLM self-check with reflection tags (`[Retrieve], [ISREL], [ISSUP], [ISUSE]` in **Self-RAG**), summarize whole datasets into hierarchical Leiden communities for big-picture questions (**Microsoft GraphRAG Global vs. Local Search**), and read charts/pages directly (**Multimodal RAG / ColPali**).

6. **[Module 6 — Autonomous Teams, Grading Everything & Explaining Why](Module_6_GraphRAG_Evaluation_and_Explainability.md):**
   Finally, we break complex questions down (**Query Decomposition & KG Paths**), give AI agents tools (**ReAct**, Web/Cypher/APIs via **MCP**) and teamwork (**Multi-Agent LangGraph/CrewAI via A2A**), scientifically prove accuracy using **Cranfield/TREC**, **Cohen's $\kappa$**, significance tests, **P@K, Recall@K, MAP, MRR, NDCG**, and **RAGAS** (Faithfulness, Answer Relevancy, Context Precision/Recall), and make every result trustworthy through **Explainable IR** (Graph paths, Enterprise provenance/ACLs, Geospatial maps, Multimedia boxes/timestamps, and Distributed IR routing).

---

## 2. Master Formula Sheet (Must-Know Equations)

1. **IDF & TF-IDF:** $\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$, $\quad w_{t,d} = \text{TF}(t,d) \times \text{IDF}(t)$
2. **Cosine Similarity:** $\cos(\theta) = \frac{\mathbf{d} \cdot \mathbf{q}}{\|\mathbf{d}\|\|\mathbf{q}\|}$
3. **BIM (Probabilistic Model):**
   $$p_i = \frac{r_i + 0.5}{|R| + 1}, \quad q_i = \frac{(n_i - r_i) + 0.5}{(N - |R|) + 1}, \quad c_i = \log_2\frac{p_i(1-q_i)}{q_i(1-p_i)}, \quad \text{RSV}(D) = \sum_{i \in Q \cap D} c_i$$
4. **BM25:**
   $$\text{Score}(D,Q) = \sum_{i} \text{IDF}(q_i) \cdot \frac{f(q_i,D)(k_1 + 1)}{f(q_i,D) + k_1\left(1 - b + b\frac{|D|}{\text{avgdl}}\right)} \qquad (k_1 \approx 1.2\text{--}2.0,\ b \approx 0.75)$$
5. **ColBERT Late Interaction:** $\text{Score}(Q,D) = \sum_{i} \max_{j} (q_i \cdot d_j)$
6. **Reciprocal Rank Fusion (RRF):** $\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)} \quad (k = 60)$
7. **Token Jaccard Similarity:** $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$
8. **Perplexity:** $\text{PPL}(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \ln P(w_i \mid w_{<i})\right) = 2^{H_{\text{bits}}}$
9. **F1 (Harmonic Mean):** $F_1 = \frac{2PR}{P+R}$
10. **MRR:** $\text{MRR} = \frac{1}{|Q|}\sum_{q} \frac{1}{\text{rank}_{\text{first relevant}}}$
11. **DCG & NDCG:** $\text{DCG}_p = \sum_{i=1}^p \frac{2^{rel_i}-1}{\log_2(1+i)}, \qquad \text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$
12. **Cohen's Kappa:** $\kappa = \frac{P(A) - P(E)}{1 - P(E)}$
13. **RAGAS Core:**
    - $\text{Faithfulness} = \frac{\text{Supported Claims in Answer}}{\text{Total Claims in Answer}}$
    - $\text{Context Recall} = \frac{\text{GT Claims Found in Retrieved Context}}{\text{Total GT Claims}}$

---

## 3. Slide Gotchas & Corrections (Impress Your Examiner!)

| Where in Prof's PDFs | What the Slide Says / Misses | The True / Corrected Answer |
| :--- | :--- | :--- |
| **`Module 1 .pptx.pdf`, Slide 39** | Labels single-vector Bi-Encoder dense retrieval as **"EARLY INTERACTION"** | **Wrong!** Bi-Encoder has *no* token interaction until one dot product; **Cross-Encoder** is Early Interaction; **ColBERT** is Late Interaction. |
| **`Module 2_ Problems on Vector Model.pdf`, Pages 1–3** | Computes IDF with $\log_2$, switches to $\log_{10}$ (`0.18`, `0.48`) for cosine, and **omits `+ 0.0324` (`0.18 × 0.18`)** in D2 dot product (`0.4608` instead of `0.4932`), giving `0.770` instead of `0.824` | Consistent exact cosine similarities are **D2 = 0.825**, **D3 = 0.327**, **D1 = 0.080** (ranking `D2 > D3 > D1` unchanged). |
| **`Module 3.pdf`, Slide 35–36 (Eval)** | Slide 35 uses $\exp(\text{NLL})$ while Slide 36 says "$2$ raised to cross-entropy in bits"; footer says "Module 4" | Both formulas are identical when base matches ($e^{\text{nats}} = 2^{\text{bits}}$); Syllabus places this under **Module 3.3**. |
| **`Module 4` PDF, Slide 66** | Skipped in exported PDF, yet Slide 98 asks for **Token Jaccard similarity** | Restored & solved in Module 4: $J(\text{"a p j abdul kalam"}, \text{"avul pakir jainulabdeen abdul kalam"}) = 2/8 = \mathbf{0.25}$. |
| **`Information Retrieval Evaluation (1).pdf`, Slide 35 & 41** | Slide 35 says rejecting $H_0$ *"shows observation **is** a random result"*; Slide 41 prints `0.7878` instead of `0.7875` | Rejecting $H_0$ shows the result is **NOT** random luck! ($630/800 = 0.7875$). |
| **Missing PPTs in Upload** | **1.4** (GQA, Longformer, FlashAttention), **2.3** (OpenSearch, pgvector, LanceDB), **5.1–5.2** (All Module 5!), **6.1, 6.2 RAGAS, 6.3** (XIR) | All researched and fully explained in plain English inside the Module `.md` files! |

---

## 4. Lab Experiments 1–11 Rapid Viva Guide (From Syllabus Pages 7–8)

1. **Exp 1 (PDF Processing + Chunking + Vector DB):** Extract text from PDFs, split cleanly with recursive chunking, embed chunks, and store in a vector DB.
2. **Exp 2 (BM25 vs. TF-IDF):** Compare BM25 against TF-IDF—BM25 wins because it caps repeated word spam ($k_1$) and normalizes doc length ($b$).
3. **Exp 3 (Dense Semantic Search via FAISS/Pinecone):** Embed text with Sentence-Transformers (`SBERT`) and run fast vector search to match meanings without exact keywords.
4. **Exp 4 (Hybrid Search):** Combine BM25 + Dense rankings using **RRF** (`1/(60+rank)`) so you get exact codes AND synonyms together.
5. **Exp 5 (Create & Visualize KG in Neo4j):** Turn extracted `(S, P, O)` triples into nodes and arrows in Neo4j using `MERGE`.
6. **Exp 6 (Graph Schema & Cypher Queries):** Design labels/constraints and write Cypher `MATCH` queries to answer multi-hop questions.
7. **Exp 7 (Multimodal Search with CLIP):** Map images and text into one shared vector space using CLIP for text-to-image and image-to-text search.
8. **Exp 8 (RAG Pipeline with LangChain/LlamaIndex):** Connect `Document Loader → Chunker → Vector DB → Top-K Retriever → LLM` to give grounded answers.
9. **Exp 9 (Evaluate RAG via Recall@K, MRR & RAGAS):** Grade retrieval (`Recall@K`, `MRR`) plus full RAG quality (**RAGAS**: Faithfulness, Answer Relevancy, Context Precision, Context Recall).
10. **Exp 10 (OCR + LLM Invoice/Resume Parsing):** Use `PyMuPDF` / `Tesseract` OCR to read text from PDFs/scans, then use LLM + **Pydantic / Instructor** to extract clean validated JSON.
11. **Exp 11 Micro-Project (End-to-End GraphRAG Dual-Query System):** Ingest raw docs into a Neo4j Knowledge Graph (`GraphTransformer` / `KnowledgeGraphIndex`) and give users **two ways to ask**: plain English questions (GraphRAG / Text-to-Cypher) AND raw **Cypher queries** for deep connected insights!
