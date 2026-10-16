# OSR-mAP

**Open-Set Retrieval mean Average Precision (OSR-mAP)**

OSR-mAP evaluates retrieval systems in **open-set settings**, where some queries do not belong to any known class.

---

## Motivation

Standard retrieval metrics such as:

- Recall@K
- mean Average Precision (mAP)

assume that **every query has at least one valid match** in the reference database.

In real-world applications such as:

- Visual Place Recognition (VPR)
- Loop Closure Detection (LCD)
- Image Retrieval
- Localization

some queries may originate from previously unseen locations and therefore should be **rejected rather than retrieved**.

As a result, retrieval quality and unknown-query rejection are often evaluated using separate metrics, making direct comparison difficult.

---

## Key Idea

OSR-mAP combines:

- **retrieval quality** (Average Precision)
- **open-set recognition** (False Acceptance Rate)

into a single threshold-independent metric.

The metric rewards methods that:

- retrieve known observations accurately,
- reject unknown observations correctly.

---

## Input Format

The metric operates directly on retrieval results:

- `y_true` — ground-truth labels
- `ranking_labels` — ranked retrieval labels (top-K)
- `distances` — retrieval distances or anomaly scores
- `ref_set` — set of labels present in the reference database

Lower distance values indicate higher confidence.

---

## Example Labels

Integer labels may be used directly:

```text
0 -> class A
1 -> class B
2 -> class C
3 -> unknown class
```

---

## Definitions

### Known and Unknown Queries

Let:

```text
Y_ref
```

denote the set of labels present in the reference database.

Known queries:

```text
Q_kn = { q | y_q ∈ Y_ref }
```

Unknown queries:

```text
Q_un = { q | y_q ∉ Y_ref }
```

---

### Relevance

For a known query:

```text
rel(k) = 1 if ranking_labels[k] == y_true
         0 otherwise
```

---

### Average Precision

For each known query:

```text
AP(q) = mean Precision@k over relevant ranks
```

---

### Closed-Set mAP

When all queries are known:

```text
mAP = mean AP(q)
```

computed over all queries.

---

### Open-Set Acceptance

For a threshold `t`:

```text
accept(q,t) = True  if distance(q) <= t
              False otherwise
```

---

### KAP (Known-query Average Precision)

```text
KAP(t)
=
mean of:

accept(q,t) * AP(q)

over all known queries
```

Equivalent interpretation:

```text
Accepted query:
contributes AP(q)

Rejected query:
contributes 0
```

KAP therefore measures how much retrieval performance is retained after applying an open-set acceptance threshold.

---

### FAR (False Acceptance Rate)

```text
FAR(t)
=
(# accepted unknown queries)
/
(# unknown queries)
```

FAR quantifies the fraction of unknown samples incorrectly accepted as known.

---

## Final Metric

For all operating thresholds:

```text
(FAR(t), KAP(t))
```

pairs are generated.

The threshold is swept across all unique distance values.

The final metric is:

```text
OSR-mAP = AUC(KAP vs FAR)
```

or equivalently:

```text
OSR-mAP =
Area under the KAP-FAR curve
```

The area is computed using trapezoidal integration.

---

## Interpretation

Large OSR-mAP values indicate:

- strong retrieval performance,
- effective rejection of unknown queries.

A system achieves high OSR-mAP only when it simultaneously:

- retrieves known places accurately,
- maintains low false acceptance of unknown observations.

---

## Special Cases

### Closed-set

When no unknown queries exist:

```text
Q_un = ∅
```

then:

```text
OSR-mAP = mAP
```

Thus OSR-mAP naturally reduces to the standard retrieval metric.

---

### Perfect Open-Set Separation

If all unknown queries receive larger distances than all known queries:

```text
OSR-mAP = mAP
```

because unknown samples can be rejected without sacrificing retrieval performance.

---

### Unknown-only Dataset

If:

```text
Q_kn = ∅
```

then retrieval quality cannot be defined:

```text
OSR-mAP = undefined
```

---

## Minimal Example

```python
from osr_map import osr_map_score

# labels:
# 0,1 -> known
# 2   -> unknown

y_true = [
    0, 0,
    1, 1,
    2, 2
]

ranking_labels = [

    [0, 0, 1, 1],
    [0, 1, 0, 1],

    [1, 1, 0, 0],
    [1, 0, 1, 0],

    [0, 1, 0, 1],
    [1, 0, 1, 0]
]

# partially overlapping distributions
distances = [
    0.10,
    0.35,
    0.20,
    0.55,
    0.25,
    0.45
]

ref_set = {
    0,
    1
}

score = osr_map_score(
    y_true=y_true,
    ranking_labels=ranking_labels,
    distances=distances,
    ref_set=ref_set,
    plot=True
)

print("OSR-mAP:", score)
```

---

## Visualization

The metric can optionally visualize the KAP-FAR curve:

```python
osr_map_score(..., plot=True)
```

The generated plot uses:

```text
X-axis : FAR
Y-axis : KAP
```

where each point corresponds to a different acceptance threshold.

---

## Properties

OSR-mAP is:

- ✅ open-set aware
- ✅ ranking aware
- ✅ threshold independent
- ✅ interpretable
- ✅ compatible with retrieval systems
- ✅ reduces to mAP in the closed-set scenario

---

## Comparison

| Metric | Open-set | Ranking-aware | Threshold-free |
|----------|----------|----------|----------|
| mAP | No | Yes | Yes |
| AUROC | Yes | No | Yes |
| OSCR | Yes | Limited | Yes |
| OSR-mAP | Yes | Yes | Yes |

---

## Typical Applications

OSR-mAP can be used in:

- Visual Place Recognition (VPR)
- Loop Closure Detection (LCD)
- Image Retrieval
- Open-World Recognition
- Open-Set Localization
- Robotic Navigation

---

## Usage Style

```python
score = osr_map_score(
    y_true,
    ranking_labels,
    distances,
    ref_set
)
```

---

## License

MIT License

Copyright (c) 2026 Piotr Woźniak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY