# Module 4: Building & Filling a Real Knowledge Graph Automatically

> **Continuing Our Story (From Module 3 → Module 4):**  
> In Module 3, we sketched Knowledge Graphs on a whiteboard. Now let's make it real! Suppose our university asks us to build an **Academic Knowledge Graph** from thousands of messy **faculty CVs, scanned certificates, and multi-column research PDFs** so anyone can ask: *"Who teaches Knowledge Graphs and co-authored an AI paper?"*  
> To pull this off, we will solve **three real engineering challenges** in order:  
> 1. **Store & Search Fast (4.1):** How does a graph database like Neo4j follow 4–5 hops in milliseconds where SQL freezes?  
> 2. **Walk, Extract & Clean Data (4.2):** How do we query graphs (Cypher, BFS/DFS), pull entities from text, stop duplicate `"A.P.J. Kalam"` nodes (**Entity Resolution**), and force LLMs to obey strict rules (**Pydantic + Instructor**)?  
> 3. **Read Scanned PDFs & Pictures (4.3):** How do we extract clean facts from scanned certificates, tables, and invoices (**OCR + Layout AI + VLMs**)?

---

## Syllabus Checklist (Every Topic Covered)
- **4.1** Graph Databases & Architecture: concepts, architecture, storage & indexing
- **4.2** Graph querying fundamentals, traversal techniques (BFS/DFS) & pattern matching; Knowledge extraction & relationship discovery; Entity Resolution & Deduplication; Schema-guided LLM extraction (**Instructor & Pydantic**) *(+ missing Slide 66 fuzzy/Jaccard similarity restored!)*
- **4.3** Document Understanding & AI-Based Extraction: OCR and LLM-based document understanding; Information extraction from multimodal documents *(+ solved answers to ALL 12 professor questions from Slides 98–99!)*

---

## Chapter 1 (Syllabus 4.1): Inside a Graph Database Engine

### 1. Transactional Graph DBs vs. Offline Analytics Engines
- **Graph Database (OLTP, e.g., Neo4j, Memgraph):** Built for real-time apps—fast reads, writes, and live user queries.
- **Graph Compute Engine (OLAP, e.g., Pregel, Giraph, Neo4j GDS):** Built for batch number-crunching across the whole graph at once (e.g., finding the most influential researcher overall).

### 2. Why Is Neo4j 1,000× Faster Than SQL on Multi-Hop Questions?
Remember the SQL `JOIN` headache from Module 3? Let's look at what happens inside the computer:
- **In SQL (or Non-Native Graph DBs like JanusGraph on Cassandra):** To go from `Asha → Paper → Co-author → Co-author's Paper`, the database must do a **binary search inside an index table (`O(log n)`) at EVERY SINGLE HOP**. With 1 million people (~50 friends each), a 4-hop query takes **1,543 seconds (~25 minutes)** in SQL—and 5 hops never finishes!
- **In a Native Graph DB (Neo4j) — Index-Free Adjacency:** Every node file record stores the **exact disk address (pointer) of its first relationship**, and each relationship points directly to the next one in a linked chain!
  - Finding the next neighbor takes **$O(1)$ constant time** (`record_ID × fixed_record_size = exact byte offset` on disk—no index lookup per hop!).
  - **The Golden Rule:** Query speed depends **ONLY on the tiny neighborhood you visit**, NOT on whether the database holds 1,000 nodes or 1 billion nodes! That same 4-hop query takes **1.36 seconds** in Neo4j!

### 3. The 5 Layers of a Graph Database (How Neo4j Works Under the Hood)
1. **API / Drivers:** Python/Java drivers talking over the fast **Bolt** protocol or Browser UI.
2. **Query Engine:** Parses your Cypher text into a tree (AST) → Cost-based Planner picks the smartest starting index → Runtime runs the plan.
3. **Transactions & Indexes:** Keeps writes **ACID**-safe (using write locks and a crash-safe **Write-Ahead Log / WAL**) + manages Indexes.
4. **Fixed-Size Storage Files:** Separate tidy stores for **Nodes**, **Relationships**, **Properties**, and **Labels/Tokens**.
5. **Memory & Cluster Replication:** A **Page Cache in RAM** keeps the hot part of the graph in memory; a **Raft cluster** ensures a write commits only once a majority of Primary servers agree, while Secondary read replicas scale reads (using **bookmarks** so users always read their own latest writes!). *Note:* Splitting a big graph across machines (**sharding**) creates "cut edges" where crossing machines costs network round-trips.

### 4. "Find, Then Expand": 6 Types of Indexes & Reading Query Plans
**Crucial Distinction:** *Indexes are ONLY used to find your **starting anchor node(s)**. Once you land on the anchor, **Index-Free Adjacency** walks all the edges!*

| Index Type | Built On | What It Speeds Up | Example |
| :--- | :--- | :--- | :--- |
| **RANGE** | B+tree-style | `=`, `<`, `>`, `STARTS WITH` | `p.year >= 2020` |
| **TEXT** | Trigram index | `CONTAINS`, `ENDS WITH` | `p.title CONTAINS 'graph'` |
| **POINT** | Spatial index | Distance / area checks | `point.distance(c.loc, $here) < 5000` |
| **FULLTEXT** | Apache Lucene | Scored keyword & fuzzy (`~`) text search | `'knowledge AND graph~'` |
| **VECTOR** | **HNSW** (from Mod 2!) | Finding top-$k$ similar meaning vectors | `db.index.vector.queryNodes(...)` |
| **LOOKUP** | Default token map | Finding all nodes with label `:Paper` | `MATCH (n:Paper)` |

- **`EXPLAIN` vs. `PROFILE`:** `EXPLAIN` previews the plan without running it; `PROFILE` actually runs it and shows **rows** and **`db hits`** (storage work—fewer `db hits` = faster!). Read plans **bottom-up**: you want `NodeIndexSeek` at the bottom, NOT `NodeByLabelScan`!
- **5 Speed Killers to Avoid:** (1) Forgetting node labels, (2) Unbounded paths `-[*]-` (always cap like `-[*1..3]-`), (3) Unconnected `MATCH (a), (b)` Cartesian products, (4) Pasting raw values instead of `$parameters` (causes Cypher injection & slow recompiling), (5) Walking blindly through a supernode.

---

## Chapter 2 (Syllabus 4.2): Querying, Walking, Extracting & Cleaning Knowledge

### 1. Cypher Querying Essentials (Drawing Patterns With ASCII Art)
Cypher (and the new ISO **GQL** standard) looks like whiteboard arrows: `(:Node)-[:EDGE]->(:Node)`.
- **Read:** `MATCH (f:Faculty)-[:AUTHORED]->(p:Paper) WHERE p.year >= 2023 RETURN f.name, p.title ORDER BY p.year DESC LIMIT 10;`
- **Safe Create (`MERGE`, NEVER duplicate!):**
  ```cypher
  MERGE (a:Person {id: 'F-102'})
    ON CREATE SET a.name = 'Asha Rao', a.created = datetime()
    ON MATCH  SET a.lastSeen = datetime()
  MERGE (s:Skill {name: 'Knowledge Graphs'})
  MERGE (a)-[:HAS_SKILL]->(s);
  ```
  *(Always `MERGE` the two endpoint nodes on their IDs first, then `MERGE` the relationship between them! Use `DETACH DELETE` to remove a node along with all its attached edges.)*

### 2. Walking the Graph: BFS vs. DFS & Graph Algorithms
When our query searches for neighbors or paths, how does the computer walk?

| Walk Style | How It Explores (Everyday Picture) | Data Structure | Finds Shortest Hops? | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **BFS (Breadth-First)** | **Ripples in a pond:** Visit everyone 1 hop away, then 2 hops away, then 3 hops away… | **Queue (FIFO)** | **Yes!** (First visit = fewest hops) | `"Researchers within 2 hops"`, Neo4j `shortestPath()` |
| **DFS (Depth-First)** | **Exploring a maze:** Follow one tunnel all the way to a dead end, then backtrack and try the next branch. | **Stack (LIFO)** or Recursion | No guarantee | Detecting cycles, topological order, listing all possible paths |

*(Both take $O(V + E)$ time; BFS uses more memory on wide frontiers, DFS uses less memory. For **weighted** shortest paths—like distance in meters—use **Dijkstra / $A^*$** in Neo4j GDS! Watch out for **path explosion**: at 20 friends per person, 3 hops = 8,000 paths, so always limit depth & edge types!)*

- **Graph Analytics (Neo4j GDS):** **PageRank / Betweenness** (who is most central/bridging?), **Louvain / Label Propagation** (which topic clusters exist?), **Jaccard Node Similarity** (who has matching skill sets?), **FastRP / Node2Vec** (graph node embeddings).

---

### 3. Automatically Extracting Facts From Text (Knowledge Extraction)
We can't hand-type 50,000 CVs and papers into Neo4j! Let's automate extraction:

**Classical 8-Step NLP Pipeline:**
`1. Clean & Split sentences → 2. POS Tagging (noun/verb) → 3. NER (find Person, Org, Date via spaCy or BERT BIO tags) → 4. Dependency Parse → 5. Coreference ("she" → Asha Rao) → 6. Relation Extraction → 7. Entity Linking → 8. MERGE into KG`.
*(Catch: If step 3 misses a name, later steps can't fix it!)*

- **Closed IE vs. Open IE:**
  - **Closed IE** only extracts relations from your allowed list (`AUTHORED`, `TEACHES`).
  - **Open IE** grabs *any* verb phrase (`wrote`, `penned`, `is the author of`)—so you must normalize/canonicalize synonyms into one clean relation (`AUTHORED`) before loading!
- **5 Ways to Do Relation Extraction:** Hand-written rules/patterns, Supervised classifiers, **Distant Supervision** (auto-tag sentences using existing KB pairs—noisy!), Joint seq2seq triple generators (e.g., **REBEL**), and **LLM prompting**.
- **Entity Linking (Which "Kalam" is this?):** Mapping a mention in text (`"missile programme led by Kalam"`) to its exact global entry in Wikidata (`A. P. J. Abdul Kalam` vs. `Abul Kalam Azad`) using surrounding context words (`DRDO, ISRO`).

---

### 4. Stopping Duplicate Nodes (Entity Resolution & Deduplication)
Suppose Paper 1 says `"Dr. A. P. J. Abdul Kalam"`, a CV says `"A.P.J Abdul Kalam"`, and a bio says `"Avul Pakir Jainulabdeen Abdul Kalam"`. If our pipeline creates 3 separate nodes, Dr. Kalam's papers live on Node 1 and his awards live on Node 3—**multi-hop queries silently fail!**

- **Terminology:** **Entity Resolution (ER)** = deciding which records are the same real-world thing; **Deduplication** = cleaning one dataset; **Record Linkage** = matching across 2+ datasets; **Canonicalisation** = picking one official master name + saving old spellings in `aliases`. (Remember: *Entity Linking* connects text to an *external* KB like Wikidata; *ER* merges records inside *your own* data!)
- **The Speed Trap & Blocking:** Comparing every pair among 1,000,000 records requires **500 billion comparisons** ($\frac{n(n-1)}{2}$)! So we run **Blocking** first: put records into buckets using cheap keys (like `last_word + first_initial` → `"kalam|a"` vs `"azad|m"`, or **Soundex** phonetic codes, **MinHash + LSH**, or **Embedding ANN**) and only compare records inside the same bucket!
- **How We Compare Two Candidates (Restoring Missing Slide 66!):**
  1. **Exact / Rule Normalization:** Lowercase, strip punctuation & titles (`dr, prof, shri`), apply initials rules.
  2. **Fuzzy / String Similarity:**
     - **Token Jaccard Similarity:** Shared words divided by total unique words: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$.
     - **Levenshtein Edit Distance:** Minimum letter edits (insert/delete/replace) for spelling typos.
     - **Jaro-Winkler:** Gives extra score when the start of two names matches.
  3. **Probabilistic Linkage (Fellegi–Sunter):** Combines field scores into three zones: `Auto-Match`, `Auto-Reject`, or `Needs Review`.
  4. **Embedding + LLM Grey-Zone Judge:** Embed `"Name | Role | Company"` with SBERT; for tricky middle cases (like `"Sahyadri Lab Supply"` missing a GST tax ID), ask an LLM judge (with confidence threshold + human review!).
  5. **Clustering:** Group matches via connected components (beware **chaining** where `A≈B` and `B≈C` pulls `A` and `C` together even though `A≠C`!).

---

### 5. Forcing LLMs to Obey Rules: Pydantic + Instructor
Why can't we just ask a raw LLM: *"Give me JSON entities and relations"*?
Because raw LLMs chat around the JSON (`"Sure! Here you go:"`), rename keys (`"Entities"` instead of `"entities"`), invent unauthorized edges (`"WORKS_WITH"`), write `"around 2015"` instead of an integer year, or **make up people not in the text**!

**The Fix — Two Tools Working Together:**
1. **Pydantic (The Strict Contract):** General Python library (knows nothing about LLMs; usable standalone). You define a `BaseModel` class with exact allowed types (`Literal["AUTHORED", "TEACHES"]`), `extra="forbid"` (blocks invented keys), and custom Python `@field_validator` checks—for example, checking that the LLM's `evidence` quote **literally appears inside `source_text`** (stops hallucinations!).
2. **Instructor (The Strict Coach):** Wraps your OpenAI/Gemini/Anthropic client (`response_model=Extraction, max_retries=3`). It hands Pydantic's schema to the LLM, checks the answer with Pydantic, and **whenever Pydantic throws a `ValidationError`, Instructor automatically sends the exact error back to the LLM and makes it try again**!

*(Then load the validated object into Neo4j using parameterized `MERGE` and save provenance like `source` file and `extractedAt` timestamp on every edge!)*

---

## Chapter 3 (Syllabus 4.3): Reading Scanned PDFs, Tables & Images (Document AI)

What if a faculty CV or certificate is a **scanned image**, or a research paper has **two columns, complex tables, charts, and stamps**? Regular text splitters turn two columns into scrambled word soup!

### 1. Digital vs. Scanned PDFs & The OCR Pipeline
- **Smart PDF Check (PyMuPDF `fitz`):** Check *each page separately* (`len(page.get_text().strip()) >= 20`)! Born-digital pages have real selectable text; scanned pages or certificate photos get rendered at **~300 dpi** and sent to **OCR**.
- **How OCR Works (6 Steps):** `1. Scan (300 dpi) → 2. Clean up (straighten tilt, remove noise, black/white) → 3. Find text boxes → 4. Read characters (LSTM/Transformer, e.g., Tesseract, PaddleOCR, docTR, AWS Textract) → 5. Spell/confidence check → 6. Output text + coordinates + confidence`.

### 2. Understanding Layout, Tables & Visuals (VLMs + Multimodal Fusion)
- **Layout & Tables:** **LayoutLM** reads each word *plus* its `(x, y)` page box so it never mixes Left Column with Right Column. **Table Transformer** (`PubTables-1M`) and `pdfplumber` rebuild rows, columns, and cells cleanly. Cross-field Pydantic validators even double-check invoice math (`qty × price == subtotal`)!
- **Vision-Language Models (VLMs):** Models like **GPT-4o, Gemini, Claude, Qwen-VL, LLaVA** (and OCR-free **Donut**) take the **page image itself** (`ViT image encoder → Projector → LLM decoder`) and read charts, stamps, seals, and handwriting directly!
- **Best Practice — Multimodal Fusion Architecture:** Don't rely on just one! Run **OCR/text layer** (exact IDs/numbers) + **Layout/Table detectors** + **VLM** (visuals/charts/stamps) in parallel → merge them with **page number + bounding-box coordinates** → extract via **LLM + Instructor + Pydantic** → send low-confidence or failed validations to a **Human Review Queue**!
- **Multimodal KG Bridge:** Extracted figures, certificate scans, and audio clips can also be saved as `:Image` / `:Audio` nodes linked straight to `:Paper` or `:Faculty` nodes!

---

## Answers to All 12 Professor Questions (Slides 98–99)

1. **Define the Property Graph model & 4 building blocks:** Stores connected data as **Nodes** (entities), **Labels** (type tags on nodes), **Relationships** (named, directed edges between two nodes), and **Properties** (typed key-value pairs on nodes or edges).
2. **Explain Index-Free Adjacency & why 3-hop cost is independent of total graph size:** Every node stores direct disk pointers to its attached relationships, which point straight to neighbor nodes. Walking an edge takes $O(1)$ pointer arithmetic instead of scanning an index of all $N$ nodes, so 3-hop speed depends only on the size of the traversed neighborhood, not the whole database.
3. **Compare BFS vs. DFS:** BFS uses a FIFO queue, visits level-by-level, guarantees shortest unweighted paths, uses $O(\text{frontier})$ memory, and powers $k$-hop neighborhoods / `shortestPath()`. DFS uses a LIFO stack/recursion, goes deep down one branch before backtracking, doesn't guarantee shortest paths, uses $O(\text{depth})$ memory, and suits cycle detection, topological sort, and listing all paths. Both run in $O(V+E)$ time.
4. **Distinguish Entity Linking vs. Entity Resolution with 1 example each:** *Entity Linking* maps a text mention to an *external* Knowledge Base entry (e.g., mapping `"Kalam"` in an article to Wikidata's `A. P. J. Abdul Kalam`). *Entity Resolution* merges duplicate records within/across *your own* datasets (e.g., merging `"Dr. A. P. J. Abdul Kalam"` from a CV and `"A.P.J Abdul Kalam"` from HR records).
5. **Cypher for faculty who teach a course and have skill `"Knowledge Graphs"`:**
   ```cypher
   MATCH (f:Faculty)-[:TEACHES]->(c:Course),
         (f)-[:HAS_SKILL]->(:Skill {name: 'Knowledge Graphs'})
   RETURN f.name AS faculty, collect(DISTINCT c.code) AS courses;
   ```
6. **Idempotent Cypher loading `(Person)-[:HAS_CERTIFICATION]->(Certification)`:**
   ```cypher
   MERGE (p:Person {id: $personId})
   ON CREATE SET p.name = $personName
   MERGE (cert:Certification {id: $certId})
   ON CREATE SET cert.title = $title, cert.issuer = $issuer
   MERGE (p)-[r:HAS_CERTIFICATION]->(cert)
   SET r.date = $date, r.source = $sourceDoc;
   ```
7. **Token Jaccard similarity of `"a p j abdul kalam"` and `"avul pakir jainulabdeen abdul kalam"`:**
   - $A = \{\text{a}, \text{p}, \text{j}, \text{abdul}, \text{kalam}\}$ (5 words) | $B = \{\text{avul}, \text{pakir}, \text{jainulabdeen}, \text{abdul}, \text{kalam}\}$ (5 words)
   - Overlap $A \cap B = \{\text{abdul}, \text{kalam}\}$ (2 words) | Union $A \cup B = 8$ unique words
   - **$J(A, B) = \frac{2}{8} = \mathbf{0.25}$** *(Shows why naive word overlap fails on initials and needs an initials rule or embeddings!)*
8. **Pydantic vs. Instructor (can either work alone?):** Pydantic defines schemas and validates Python data (knows nothing about LLMs; **yes**, works standalone). Instructor wraps LLM APIs to pass Pydantic schemas and retry on `ValidationError` (**no**, requires Pydantic).
9. **`PROFILE` shows `NodeByLabelScan` over 2M `:Person` nodes + `Filter`. Fix:** Missing property index! Run `CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);` and query with `MATCH (p:Person {name: $name})` so Neo4j uses fast `NodeIndexSeek`.
10. **Vendor deduplication (`R1` & `R2` share GSTIN `27...`; `R3` missing GSTIN + uses `gmail.com`):** Block on `gstin` and `first_word + city` (`"sahyadri|pune"`). Rule auto-merges `R1 ↔ R2` (exact GSTIN). Send grey-zone pair `(R1, R3)` to an LLM judge (`MatchDecision` via Instructor) and auto-merge only above a confidence threshold, routing uncertain cases to human review.
11. **LLM returns `"WORKS_WITH"` and a hallucinated person:** `Literal[...]` / `Enum` rejects invalid relation `"WORKS_WITH"`; a custom `@field_validator("evidence")` checking `v in info.context["source_text"]` catches the hallucinated person!
12. **OCR+LLM vs. VLM-only on invoices + Certificate model + 10k-CV pipeline:** OCR+LLM is cheaper, accurate on clean characters, and gives word bounding boxes, but struggles with stamps/charts/messy tables; VLM sees full visual layout and stamps, but costs more and can misread tiny digits (hybrid fusion wins!). Certificate schema: `(:Person)-[:HAS_CERTIFICATION {grade, date, source, bbox}]->(:Certification)-[:ISSUED_BY]->(:Organization)`. 10k-CV pipeline: Per-page digital/OCR check + VLM → LLM + Instructor + Pydantic → Blocking + ER → Confidence routing (human review for low confidence) → Idempotent Neo4j `MERGE` with source provenance.

---

## Where We Go Next → Module 5
We now have a clean Vector Store (Module 2) AND a clean Knowledge Graph filled from text and scanned documents (Module 4)! So we plug them into an LLM—and boom, we have **Basic RAG**. **Wait:** the very next day, users start complaining that Basic RAG cuts sentences in half, misses company context, pulls irrelevant docs, fails big-picture summary questions, and ignores charts! **Module 5** fixes every single one of those RAG failures!
