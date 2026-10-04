import matplotlib.pyplot as plt
import numpy as np

labels = [
    "Paul15 / Linear SVM",
    "Paul15 / Logistic regression",
    "PBMC3K / Linear SVM",
    "PBMC3K / Logistic regression",
]

random_vals = [1.63, 1.37, 0.37, 0.00]
nearest_vals = [11.33, 9.33, 4.67, 1.33]
same_class_vals = [17.39, 15.00, 5.67, 1.00]

x = np.arange(len(labels))
w = 0.24

fig, ax = plt.subplots(figsize=(3.45, 2.25))

ax.bar(x - w, random_vals, width=w, label="Random")
ax.bar(x, nearest_vals, width=w, label="Nearest")
ax.bar(x + w, same_class_vals, width=w, label="Same class")

ax.set_ylabel("Target flip rate (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", linewidth=0.5, alpha=0.35)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.tight_layout(pad=0.5)
fig.savefig("figure1_structured_vs_random_compact.pdf", bbox_inches="tight")
fig.savefig("figure1_structured_vs_random_compact.png", dpi=300, bbox_inches="tight")
plt.close(fig)