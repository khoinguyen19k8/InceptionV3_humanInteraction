import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

MODEL_PATH = "weights-and-models"

pos_list = [2, 3, 4, 6, 7, 8, 9, 10]

metrics_arr = np.zeros(
    (9, 4, 3)
)  # 9 frame values, 4 classes, 3 metrics(precision, recall, f1)
accuracy_auc_arr = np.zeros((9, 2))  # 9 frame values, 2 metrics(accuracy, auc)

for pos_val in pos_list:
    MODEL_PATH = os.path.join("weights-and-models", f"frame_val_{pos_val}")
    ARRAY_PATH = os.path.join("preprocessed-arrays", f"frames_pos_{pos_val}")

    json_file = open(os.path.join(MODEL_PATH, "model.json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(os.path.join(MODEL_PATH, "weights.best.hdf5"))
    print(f"Loaded model number {pos_val} from disk")

    X_test = np.load(os.path.join(ARRAY_PATH, "x_test.npy"))
    Y_test = np.load(os.path.join(ARRAY_PATH, "y_test.npy"))

    loaded_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )

    y_pred_proba = loaded_model.predict(X_test)

    label_pred = np.argmax(y_pred_proba, axis=1)
    label_true = np.argmax(Y_test, axis=1)

    result = classification_report(label_true, label_pred)
    auc_score = roc_auc_score(label_true, y_pred_proba, multi_class="ovr")
    accuracy_score = result["accuracy"]

    for label in [0, 1, 2, 3]:
        for i, metric in enumerate(["precision", "recall", "f1-score"]):
            metrics_arr[pos_val - 2, label, i] = result[str(label)][metric]

    accuracy_auc_arr[pos_val - 2, :] = np.array([accuracy_score, auc_score])

np.save(os.path.join("preprocessed-arrays", "metrics_arr.npy"), metrics_arr)
np.save(os.path.join("preprocessed-arrays", "accuracy_auc_arr.npy"), accuracy_auc_arr)
