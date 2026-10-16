import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from osr_map import osr_map_score

# =====================================================
# EXAMPLES GENERATED USING osr_map_score()
# =====================================================

examples = [

    # =================================================
    # HIGH QUALITY
    # =================================================
    {
        "name": "High-quality",
        "color": "green",
        "closed_map": 0.92,

        "y_true":
            [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],

        "ranking_labels": [

            [0,0,1,2],
            [0,0,2,1],
            [0,1,0,2],
            [0,0,1,2],

            [1,1,0,2],
            [1,0,1,2],
            [1,1,2,0],
            [1,0,1,2],

            [2,2,0,1],
            [2,0,2,1],
            [2,2,1,0],
            [2,0,2,1],

            [0,1,2,0],
            [1,2,0,1],
            [2,1,0,2],
            [0,2,1,0]
        ],

        "distances": [
            0.05, 0.08, 0.10, 0.14,
            0.07, 0.11, 0.16, 0.20,
            0.09, 0.13, 0.18, 0.24,

            0.15, 0.28, 0.45, 0.80
        ]
    },

    # =================================================
    # MEDIUM QUALITY
    # =================================================
    {
        "name": "Medium-quality",
        "color": "orange",
        "closed_map": 0.75,

        "y_true":
            [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],

        "ranking_labels": [

            [0,1,0,2],
            [0,0,1,2],
            [1,0,0,2],
            [0,2,1,0],

            [1,0,1,2],
            [1,1,0,2],
            [0,1,1,2],
            [1,2,0,1],

            [2,0,2,1],
            [2,2,0,1],
            [0,2,2,1],
            [2,1,0,2],

            [0,1,2,0],
            [1,0,2,1],
            [2,0,1,2],
            [0,2,1,0]
        ],

        "distances": [
            0.05,0.10,0.16,0.20,
            0.12,0.18,0.22,0.28,
            0.14,0.23,0.30,0.38,

            0.15,0.25,0.42,0.65
        ]
    },

    # =================================================
    # LOW QUALITY
    # =================================================
    {
        "name": "Low-quality",
        "color": "red",
        "closed_map": 0.55,

        "y_true":
            [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],

        "ranking_labels": [

            [1,0,2,0],
            [0,1,2,0],
            [2,0,1,0],
            [1,2,0,0],

            [0,1,2,1],
            [2,1,0,1],
            [1,0,2,1],
            [0,2,1,1],

            [0,2,1,2],
            [1,2,0,2],
            [2,0,1,2],
            [1,0,2,2],

            [0,1,2,0],
            [1,0,2,1],
            [2,1,0,2],
            [0,2,1,0]
        ],

        "distances": [
            0.10,0.15,0.20,0.25,
            0.18,0.22,0.28,0.35,
            0.25,0.32,0.40,0.50,

            0.12,0.18,0.22,0.30
        ]
    }
]

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(11, 8))

for ex in examples:

    result = osr_map_score(
        y_true=ex["y_true"],
        ranking_labels=ex["ranking_labels"],
        distances=ex["distances"],
        ref_set={0, 1, 2},
        return_curve=True,
        plot=False
    )

    print(
        f"{ex['name']:15s}"
        f"  OSR-mAP = {result['score']:.3f}"
    )

    # OSR-mAP curve
    plt.plot(
        result["far"],
        result["kap"],
        marker="o",
        markersize=8,
        linewidth=3,
        color=ex["color"],
        label=f"{ex['name']} OSR-mAP = {result['score']:.3f}"
    )

    # Area under curve
    plt.fill_between(
        result["far"],
        result["kap"],
        color=ex["color"],
        alpha=0.12
    )

    # Corresponding closed-set mAP
    plt.axhline(
        ex["closed_map"],
        color=ex["color"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"{ex['name']} closed-set mAP = {ex['closed_map']:.2f}"
    )

# =====================================================
# IDEAL SYSTEM
# =====================================================

plt.plot(
    [0, 0, 1],
    [0, 1, 1],
    "k:",
    linewidth=4,
    label="Ideal OSR-mAP = 1.00"
)

# =====================================================
# STYLE
# =====================================================

plt.xlabel(
    "False Acceptance Rate (FAR)",
    fontsize=15
)

plt.ylabel(
    "Known-query Average Precision (KAP)",
    fontsize=15
)

plt.title(
    "Relationship Between Closed-set mAP and Open-set OSR-mAP",
    fontsize=17
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(True, alpha=0.3)

plt.legend(
    loc="lower right",
    fontsize=12
)

plt.tight_layout()
plt.show()
