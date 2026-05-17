
# OSR-mAP

**Open-Set Retrieval mean Average Precision (OSR-mAP)**

OSR-mAP evaluates retrieval systems in **open-set settings**, where some queries do not belong to any known class.

---

## Motivation

Standard metrics like:

- Recall@k  
- mean Average Precision (mAP)

assume that **every query has a valid match**.

In real-world applications (e.g. loop closure detection, visual place recognition), some queries are **unknown** and should be rejected.

---

## Key Idea

OSR-mAP combines:

- **ranking quality** (Average Precision)  
- **unknown rejection** (False Acceptance Rate)  

into a **single, threshold-independent metric**.

---

## Input Format

The metric operates directly on retrieval results:

- `y_true` — ground-truth labels  
- `ranking_labels` — ranked predictions (top-K labels)  
- `distances` — confidence scores (lower = more confident)  
- `ref_set` — set of known labels  

---

## Example Labels

You can use **integer labels**:

```text
0 → class A  
1 → class B  
2 → unknown class
````

***

## Definitions

### Relevance

```text
rel(k) = 1 if ranking_labels[k] == y_true else 0
```

***

### Average Precision

```text
AP(q) = mean Precision@k over relevant positions
```

***

### Open-set Decision

```text
accept(q, t) = True if distance(q) <= t
```

***

### KAP (Known AP)

```text
KAP(t) = mean of AP(q) over accepted known queries
```

***

### FAR (False Acceptance Rate)

```text
FAR(t) = (# unknown queries accepted) / (# unknown queries)
```

***

## Final Metric

```text
OSR-mAP = area under KAP vs FAR curve
```

***

## Special Cases

Closed-set:

```text
OSR-mAP = mAP
```

Unknown-only:

```text
OSR-mAP = undefined
```

***

## ✅ Minimal Example (Functional API)

```python
from osr_map import osr_map_score

# ground truth (0=A, 1=B, 2=unknown)
y_true = [0, 0, 1, 1, 2, 2]

# ranking (top-k)
ranking_labels = [
    [0, 0, 1, 1],  # good match
    [0, 1, 0, 1],  # slightly worse
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],  # unknown → mixed
    [1, 0, 1, 0]
]

# distances (confidence inverse)
distances = [0.1, 0.3, 0.2, 0.4, 0.9, 1.1]

# known classes
ref_set = {0, 1}

score = osr_map_score(
    y_true=y_true,
    ranking_labels=ranking_labels,
    distances=distances,
    ref_set=ref_set,
    plot=True
)

print("OSR-mAP:", score)
```

***

## Plot

The metric can optionally visualize the curve:

```python
osr_map_score(..., plot=True)
```

This plots:

* X-axis: FAR (False Acceptance Rate)
* Y-axis: KAP (Known AP)

***

## Properties

OSR-mAP is:

* ✅ open-set aware
* ✅ ranking-aware
* ✅ threshold-independent
* ✅ simple to use (function API like `accuracy_score`)

***

## Comparison

| Metric  | Open-set | Ranking | Threshold-free |
| ------- | -------- | ------- | -------------- |
| mAP     | no       | yes     | yes            |
| AUROC   | yes      | no      | yes            |
| OSCR    | yes      | partial | yes            |
| OSR-mAP | yes      | yes     | yes            |

***

## Usage Style

```python
score = osr_map_score(y_true, ranking_labels, distances, ref_set)
```

***

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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

