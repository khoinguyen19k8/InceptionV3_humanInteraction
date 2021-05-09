import matplotlib.pyplot as plt
import numpy as np
import os
import sys

metrics_arr = np.load(
    os.path.join("preprocessed-arrays", "metrics_arr.npy")
)  # 9 frame values, 4 classes, 3 metrics(precision, recall, f1)
accuracy_auc_arr = np.load(
    os.path.join("preprocessed-arrays", "accuracy_auc_arr.npy")
)  # 9 frame values, 2 metrics(accuracy, auc)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
colors = ["red", "blue", "green", "orange"]
labels = ["Hand shake", "Kiss", "High five", "Hug"]
frame_range = np.arange(2, 11)

PLOT_PATH = "plots"

if not os.path.isdir(PLOT_PATH):
    os.mkdir(PLOT_PATH)


def precision_plot(colors, labels, x_range, ax):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(x_range, metrics_arr[x_range - 2, i, 0], label=label, color=color)
    ax.set_xlabel("Frame values")
    ax.set_ylabel("Precision")
    ax.set_title("Precision for each interaction")
    ax.legend()


def recall_plot(colors, labels, x_range, ax):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(x_range, metrics_arr[x_range - 2, i, 1], label=label, color=color)
    ax.set_xlabel("Frame values")
    ax.set_ylabel("Recall")
    ax.set_title("Recall for each interaction")
    ax.legend()


def f1_plot(colors, labels, x_range, ax):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(x_range, metrics_arr[x_range - 2, i, 2], label=label, color=color)
    ax.set_xlabel("Frame values")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 score for each interaction")
    ax.legend()


def accuracy_plot(colors, labels, x_range, ax):
    ax.plot(x_range, accuracy_auc_arr[x_range - 2, 0])

    ax.set_xlabel("Frame values")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model accuracy for each frame value")


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(PLOT_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


precision_plot(colors, labels, frame_range, axes[0, 0])
recall_plot(colors, labels, frame_range, axes[0, 1])
f1_plot(colors, labels, frame_range, axes[1, 0])
accuracy_plot(colors, labels, frame_range, axes[1, 1])

save_fig(sys.argv[1])
