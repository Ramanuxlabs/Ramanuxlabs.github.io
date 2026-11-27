# RegulationQA Explorer

3D explorer for the RegulationQA benchmark and the retrieval outputs of
*Adaptive Span Extraction with Hybrid Retrieval for Regulatory Question Answering* (Chowdhury & Dey, CODS 2026).

What the graph shows
- Orange hubs: the 20 indexed instruments. Passages (500-word chunks, 766 in all) are coloured by instrument family.
- Grey links: ±1 window neighbours (consecutive chunks of one document, the window the pipeline reads).
- Orange links: BGE cosine similarity >= 0.80 between passages outside each other's window (top 3 per passage), from the released FAISS index.
- Violet nodes: the 1,002 questions, each linked to the passage its top-1 span came from. Each question is also linked to its source instrument(s); those links (and passage-to-instrument links) appear when something is selected and are used for path finding.
- Click a question to see the reference answer, the returned span inside its passage, and its top-5 candidates highlighted in the graph.

Files
- `index.html`, `graph.json`, `3d-force-graph.min.js` (v1.80.2, MIT, vendored so the page has no CDN dependency).
- `make_graph.py` rebuilds `graph.json`: `python make_graph.py /path/to/RegulationQA_dataset graph.json` (needs numpy, pandas, faiss-cpu).

Deploy on GitHub Pages: copy the folder into the repo (e.g. `docs/`), keep the `.nojekyll` file, enable Pages for that folder.
Deep link: `.../#q-207` opens question 207. Local test: `python -m http.server`, then http://localhost:8000.

## Live query
Type a question in the Live query bar (or press "Run this question live" on any question) and the page runs the pipeline in the browser, then replays it on the graph in six steps:
1. BM25 top 100 (yellow) - exact port of the released `bm25.pkl` scoring (verified identical scores on 42 test queries).
2. BGE top 100 (blue; white = in both) - query embedded in the browser, compared with the released FAISS vectors (`vectors.f16`).
3. RRF (k = 60) -> pool of 20, linked to a query node.
4. ±1 window neighbours (light blue).
5. Cross-encoder scores every sentence of the 20 passages; candidates re-ranked by their best own sentence.
6. The best sentence in the top candidate's window is returned, shown in context (with the reference answer and both F1s when the query is a RegulationQA question).

Models are loaded with transformers.js 4.3.2 from jsDelivr / Hugging Face: `Xenova/bge-base-en-v1.5` and `Xenova/ms-marco-MiniLM-L-6-v2`, 8-bit quantised, about 135 MB on first use and cached by the browser afterwards. Quantisation and the browser sentence splitter (Intl.Segmenter instead of NLTK) mean live results can differ slightly from the released Python run. If the models cannot be downloaded, the page still runs the BM25 stages.
