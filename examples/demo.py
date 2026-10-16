import matplotlib
matplotlib.use("TkAgg")

from osr_map import osr_map_score

# ==========================================
# 0,1,2 = known classes
# 3 = unknown
# ==========================================

y_true = [
    0,0,0,0,
    1,1,1,1,
    2,2,2,2,
    3,3,3,3,3,3,3,3
]

ranking_labels = [

    # class 0
    [0,0,1,2],
    [0,1,0,2],
    [1,0,0,2],
    [0,2,1,0],

    # class 1
    [1,1,0,2],
    [1,0,1,2],
    [0,1,1,2],
    [1,2,0,1],

    # class 2
    [2,2,0,1],
    [2,0,2,1],
    [0,2,2,1],
    [2,1,0,2],

    # unknowns
    [0,1,2,0],
    [1,0,2,1],
    [2,1,0,2],
    [0,2,1,0],
    [1,2,0,1],
    [2,0,1,2],
    [0,1,0,2],
    [1,2,1,0],
]

distances = [

    # known
    0.05,
    0.12,
    0.18,
    0.25,

    0.08,
    0.16,
    0.22,
    0.32,

    0.10,
    0.20,
    0.28,
    0.40,

    # unknown
    0.15,
    0.24,
    0.35,
    0.50,
    0.60,
    0.75,
    0.90,
    1.10
]

ref_set = {0,1,2}

result = osr_map_score(
    y_true=y_true,
    ranking_labels=ranking_labels,
    distances=distances,
    ref_set=ref_set,
    return_curve=True,
    plot=True
)

print("\n===== RESULT =====")
print("Mode    :", result["mode"])
print("OSR-mAP :", result["score"])

print("\nFAR")
print(result["far"])

print("\nKAP")
print(result["kap"])