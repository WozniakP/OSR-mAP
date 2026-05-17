from osr_map import osr_map_score

# ground truth (0=A, 1=B, 2=unknown)
y_true = [0, 0, 1, 1, 2, 2]

# ranking (top-k labels)
ranking_labels = [
    [0, 0, 1, 1],  # good match for 0
    [0, 1, 0, 1],  # slightly worse 0
    [1, 1, 0, 0],  # good match for 1
    [1, 0, 1, 0],  # slightly worse 1

    [0, 1, 0, 1],  # unknown → mixed
    [1, 0, 1, 0]   # unknown → mixed
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
