# Typed Decisions Official `test` Evaluation Ledger

Typed Decisions `test` (`400` cases, `2,000` decisions) may be evaluated **at most three more times** under `plan.md`.
`bench/eval_public_benchmarks.py --suite typed_decisions --ledger-entry <ID>` refuses to run unless `<ID>` has `Status = READY`, `Test Acc = PENDING`, and `Git SHA` matching `git rev-parse HEAD`.

| ID | Git SHA | Status | OOF Acc | OOF KL | Test Acc | Notes |
|---|---|---|---|---|---|---|
| hist-v1.2.1 | 43cd3a296bbc858189d1e183eaed6e89f95e5dfd | HISTORICAL | n/a | n/a | 0.7540 | DeBERTa-v3-large v1.2.1 fitted cross-encoder |
| hist-v1.3.0 | 51acf9e097ade3fb1068b3511490ef2756485927 | HISTORICAL | 0.7988 | 0.0879 | 0.7900 | ModernBERT-large 8K tree-isolated parallel-span v1.3.0 |
| slot-1 | PENDING | EMPTY | PENDING | PENDING | PENDING | Reserved for Phase 1 release candidate |
| slot-2 | PENDING | EMPTY | PENDING | PENDING | PENDING | Reserved for Phase 2 joint JevBench + Typed Decisions RC |
| slot-3 | PENDING | EMPTY | PENDING | PENDING | PENDING | Emergency / final release candidate reserve |
