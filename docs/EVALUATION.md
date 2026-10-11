# Retrieval and grounding evaluation

## Reproduce

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe evaluation\run_retrieval.py
```

The script writes actual computed metrics to
[`evaluation/results.json`](../evaluation/results.json). It uses the configured
sentence-transformer model; its weights must be cached locally or downloadable.
It does not call an LLM or a vision provider.

The regular offline API tests are run with:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m pytest -q tests\test_document_pipeline.py
```

Those tests replace embeddings with a deterministic test-only encoder so they
do not need model downloads or external credentials.

## Dataset and labels

[`evaluation/dataset.json`](../evaluation/dataset.json) contains six short
synthetic chunks, five answerable questions with manually checked relevant
chunk IDs, and two intentionally unsupported questions. Each answerable
question is labeled with one or more relevant passages based on the supplied
text. This is a small reproducibility check, not an independently curated
document benchmark.

## Retrieval methods

The script evaluates the exact same five answerable queries with:

1. Dense semantic retrieval using the configured embedding model and normalized
   FAISS inner-product similarity.
2. Hybrid retrieval, combining the dense ranking and local BM25 ranking using
   Reciprocal Rank Fusion with `k=60`.

There is no cross-encoder reranker in this implementation; therefore no
reranker scores or reranking comparison are reported.

## Metrics

- Recall@k: fraction of relevant chunks recovered in the top k.
- Precision@k: relevant retrieved chunks divided by k.
- MRR: reciprocal rank of the first relevant chunk.
- nDCG@k: graded ranking metric with binary relevance labels.
- Offline grounding proxy: whether top retrieved evidence intersects a
  manually relevant chunk, whether constructed citation IDs belong to the
  stored dataset, and whether the dense relevance threshold abstains on the
  two unsupported questions.

The grounding proxy does not score generated-answer entailment or linguistic
quality. Citation record integrity is a structural check; it does not prove
that a cited passage supports each individual claim.

## Results

Results are produced by running the command above and written to
[`evaluation/results.json`](../evaluation/results.json). The JSON records the
model name, dataset size, per-query metrics, aggregate metrics, the RRF
configuration, and limitations. Scores must be regenerated if the model,
dataset, threshold, or retrieval algorithm changes; no results should be
copied into reports unless the file was generated from that exact code and
configuration.

The recorded run used `sentence-transformers/all-MiniLM-L6-v2` on six chunks
and five answerable queries. Dense and hybrid results were identical:

| Method | MRR | Recall@1 | Recall@3 | Precision@1 | nDCG@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 |
| Hybrid (RRF) | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 |

On the two synthetic unsupported questions, both methods abstained under the
configured dense-similarity threshold (2/2). The offline citation-record
integrity check was 100% for both methods. Hybrid retrieval did **not** improve
these results, so this tiny benchmark provides no evidence of a hybrid
advantage. These perfect/high values reflect an intentionally small,
synthetic, easy-to-retrieve corpus, not expected performance on real documents.

## Limitations and interpretation

The dataset is synthetic, tiny, and mostly single-fact. It does not represent
OCR noise, long PDFs, tables, charts, domain vocabulary, ambiguous questions,
or adversarial content. Statistical conclusions are not appropriate. The
threshold is a configurable retrieval filter, not a calibrated correctness
probability. External LLM and visual model behavior is excluded from this
offline benchmark and should be evaluated separately with mocked unit tests
and clearly marked provider integration tests.
