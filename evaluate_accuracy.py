import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_recall_fscore_support,
    roc_curve, auc, PrecisionRecallDisplay,
    precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from app.utils import precompute_embeddings_for_images, load_embedding_from_db
from app.models import Student
from app import create_app, db

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

app = create_app()

def load_labels(csv_path, base_dir=None):
    """Load (image_path, usn) from CSV optionally prepending base dir."""
    labels = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) == 2:
                path, usn = row[0].strip(), row[1].strip().upper()
                if base_dir:
                    path = os.path.join(base_dir, path)
                labels.append((path, usn))
    return labels

def run_eval_with_cached_embeddings(dataset, image_embeddings, known_embeddings, threshold):
    y_true, y_pred = [], []
    norm_gallery = {k: (v / (np.linalg.norm(v) + 1e-10)) for k, v in known_embeddings.items()}
    for img_path, true_usn in dataset:
        if img_path not in image_embeddings:
            print(f"Missing or no valid face in {img_path}")
            continue
        emb = image_embeddings[img_path]
        best_usn, best_sim = None, -1.0
        for usn, g_emb in norm_gallery.items():
            sim = float(np.dot(emb, g_emb))
            if sim > best_sim:
                best_sim, best_usn = sim, usn
        pred_usn = best_usn if best_sim >= threshold else "UNKNOWN"

        y_true.append(true_usn)
        y_pred.append(pred_usn)
    return y_true, y_pred

with app.app_context():
    # Load augmented dataset labels (train+augmented)
    
    augmented_base_dir = "instance_augmented/train"
    augmented_csv = os.path.join(augmented_base_dir, "augmented_labels.csv")
    augmented_labels = load_labels(augmented_csv, base_dir=augmented_base_dir)

    # Load manually created test dataset labels
    test_csv = os.path.join(app.static_folder, "test_labels.csv")
    test_base_dir = app.static_folder  # typically static/test_images
    test_labels = load_labels(test_csv, base_dir=test_base_dir)

    # Combine augmented dataset labels with manual test set for final testing
    combined_labels = augmented_labels + test_labels

    # Stratified split of augmented dataset into validation subset for threshold tuning
    paths, labels = zip(*augmented_labels)
    val_paths, _, val_labels, _ = train_test_split(
        paths, labels, test_size=0.8, random_state=42, stratify=labels
    )
    val_set = list(zip(val_paths, val_labels))

    # The combined labels (augmented+manual test) are used as test set
    test_set = combined_labels

    # Limit validation subset size for faster threshold tuning
    VAL_SUBSET_SIZE = 100
    val_set_subset = val_set if len(val_set) <= VAL_SUBSET_SIZE else random.sample(val_set, VAL_SUBSET_SIZE)

    # Load known embeddings from DB normalized
    students = Student.query.all()
    known_embeddings = {}
    for student in students:
        emb = load_embedding_from_db(student)
        if emb is not None and emb.size > 0:
            known_embeddings[student.usn.strip().upper()] = emb / np.linalg.norm(emb)

    # Precompute embeddings for validation subset and combined test set images
    val_image_paths = [p for p, _ in val_set_subset]
    test_image_paths = [p for p, _ in test_set]

    from app.utils import mtcnn, resnet  # Use your initialized model instances

    val_image_embeddings = precompute_embeddings_for_images(val_image_paths, min_prob=0.95)
    test_image_embeddings = precompute_embeddings_for_images(test_image_paths, min_prob=0.95)

    # Threshold candidates for tuning
    threshold_candidates = np.linspace(0.3, 0.95, 14)
    best_thr, best_acc = 0.7, -1
    accs = []

    # Threshold tuning on validation subset
    for thr in threshold_candidates:
        yt, yp = run_eval_with_cached_embeddings(val_set_subset, val_image_embeddings, known_embeddings, thr)
        if yt:
            acc = accuracy_score(yt, yp)
            accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        else:
            accs.append(0)

    print(f"Best threshold: {best_thr:.3f} with accuracy {best_acc:.4f}")

    # Plot accuracy vs threshold
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_candidates, accs, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.grid(True)
    plt.savefig(os.path.join(augmented_base_dir, "threshold_accuracy.png"))

    # Evaluate on combined test set with best threshold
    y_true, y_pred = run_eval_with_cached_embeddings(test_set, test_image_embeddings, known_embeddings, best_thr)

    labels = sorted(set(known_embeddings.keys()).union({"UNKNOWN"}))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(augmented_base_dir, "confusion_matrix.png"))

    # Print and save classification report
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    with open(os.path.join(augmented_base_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Per-class Precision, Recall, F1-score bar plot
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, prec, width, label="Precision")
    plt.bar(x, rec, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.xticks(x, labels, rotation=90)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, F1-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(augmented_base_dir, "prf_per_class.png"))

    print("Evaluation completed. Visualizations saved.")
