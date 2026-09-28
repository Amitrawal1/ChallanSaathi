Retrieval evaluation on 48 questions (1779 chunks, top-5, state auto-detected).

| Configuration | Hit@1 | Hit@5 | MRR@5 | Missed question ids |
|---|---|---|---|---|
| Vector only | 56.2% | 77.1% | 0.640 | 2, 6, 10, 26, 27, 37, 41, 42, 43, 44, 47 |
| BM25 only | 45.8% | 70.8% | 0.557 | 1, 2, 6, 7, 9, 10, 11, 27, 31, 37, 41, 42, 43, 47 |
| BM25 only + query expansion | 56.2% | 81.2% | 0.654 | 2, 6, 10, 11, 27, 31, 37, 43, 47 |
| Hybrid (RRF) | 62.5% | 77.1% | 0.698 | 1, 2, 6, 7, 10, 37, 41, 42, 43, 44, 47 |
| Hybrid + expansion (BM25 side) | 66.7% | 81.2% | 0.740 | 2, 6, 9, 10, 37, 42, 43, 44, 47 |
| Hybrid + expansion (both sides) | 66.7% | 87.5% | 0.766 | 6, 10, 37, 43, 44, 47 |
