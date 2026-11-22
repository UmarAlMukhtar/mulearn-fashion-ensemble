"""Compare a bootstrap CNN ensemble with one CNN on the specified tiny subset."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUTPUT = Path(__file__).resolve().parent / "results"
SEED = 42
LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def dataset(images, labels, seed):
    """Keep TensorFlow's thread use small on a CPU runner."""
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(len(images), seed=seed).batch(8)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return data.with_options(options)


def train(images, labels, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(dataset(images, labels, seed), epochs=3, verbose=2)
    return model


def probabilities(model, images):
    values = model(images, training=False).numpy()
    assert values.shape == (len(images), 10) and np.isfinite(values).all()
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-5)
    return values


def accuracy(labels, values):
    return float(np.mean(labels == values.argmax(axis=1)))


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    (x_all, y_all), (x_test_all, y_test_all) = keras.datasets.fashion_mnist.load_data()
    images = x_all[:50].astype("float32")[..., None] / 255
    labels = y_all[:50]
    test_images = x_test_all[:50].astype("float32")[..., None] / 255
    test_labels = y_test_all[:50]
    train_ids, val_ids = train_test_split(np.arange(50), test_size=0.2, random_state=SEED)
    assert not np.intersect1d(train_ids, val_ids).size
    assert len(train_ids) + len(val_ids) == 50
    assert images.shape == (50, 28, 28, 1) and test_images.shape == images.shape
    assert 0 <= images.min() <= images.max() <= 1
    rng = np.random.default_rng(SEED)
    validation_members, test_members, bootstrap_records, member_results = [], [], [], []

    for member in range(5):
        bootstrap_ids = rng.choice(train_ids, size=len(train_ids), replace=True)
        assert np.isin(bootstrap_ids, train_ids).all()
        model = train(images[bootstrap_ids], labels[bootstrap_ids], SEED + member)
        val_prob = probabilities(model, images[val_ids])
        test_prob = probabilities(model, test_images)
        validation_members.append(val_prob)
        test_members.append(test_prob)
        bootstrap_records.append(bootstrap_ids.tolist())
        member_results.append({
            "member": member + 1,
            "validation_accuracy": accuracy(labels[val_ids], val_prob),
            "test_accuracy": accuracy(test_labels, test_prob),
        })

    ensemble_val = np.mean(validation_members, axis=0)
    ensemble_test = np.mean(test_members, axis=0)
    single = train(images[train_ids], labels[train_ids], seed=SEED + 100)
    single_val = probabilities(single, images[val_ids])
    single_test = probabilities(single, test_images)
    result = {
        "seed": SEED,
        "training_count": len(train_ids),
        "validation_count": len(val_ids),
        "test_count": len(test_images),
        "epochs_per_model": 3,
        "ensemble_size": 5,
        "single": {
            "validation_accuracy": accuracy(labels[val_ids], single_val),
            "test_accuracy": accuracy(test_labels, single_test),
        },
        "ensemble": {
            "validation_accuracy": accuracy(labels[val_ids], ensemble_val),
            "test_accuracy": accuracy(test_labels, ensemble_test),
        },
        "members": member_results,
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__},
    }

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUTPUT / "split-and-bootstrap.json").write_text(json.dumps({
        "train_indices": train_ids.tolist(), "validation_indices": val_ids.tolist(),
        "test_indices": list(range(50)), "bootstrap_train_indices": bootstrap_records,
    }, indent=2) + "\n")
    np.savez_compressed(OUTPUT / "probabilities.npz",
                        ensemble_validation=ensemble_val, ensemble_test=ensemble_test,
                        single_validation=single_val, single_test=single_test,
                        member_validation=np.array(validation_members),
                        member_test=np.array(test_members))
    with (OUTPUT / "test-predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "actual", "single_prediction", "ensemble_prediction"])
        writer.writerows(zip(range(50), test_labels, single_test.argmax(axis=1),
                             ensemble_test.argmax(axis=1)))

    positions = np.arange(2)
    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.bar(positions - .18, [result["single"]["validation_accuracy"],
                              result["single"]["test_accuracy"]], width=.36, label="Single CNN")
    axis.bar(positions + .18, [result["ensemble"]["validation_accuracy"],
                              result["ensemble"]["test_accuracy"]], width=.36, label="5-model ensemble")
    axis.set(xticks=positions, xticklabels=["Validation (10)", "Test (50)"],
             ylim=(0, 1), ylabel="Accuracy", title="Fashion-MNIST · first 50 records per split")
    axis.legend()
    for bars in axis.containers:
        axis.bar_label(bars, fmt=lambda value: f"{value:.0%}", padding=3)
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5), layout="constrained")
    for index, axis in enumerate(axes.flat):
        predicted = ensemble_test[index].argmax()
        axis.imshow(test_images[index, ..., 0], cmap="gray")
        axis.set_title(f"Actual: {LABELS[test_labels[index]]}\nEnsemble: {LABELS[predicted]}", fontsize=9)
        axis.set_axis_off()
    fig.suptitle("First ten held-out test images")
    fig.savefig(OUTPUT / "test-examples.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
