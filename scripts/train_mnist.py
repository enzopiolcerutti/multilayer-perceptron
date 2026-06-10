from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
from mlp import MLP
from mlp.losses import one_hot

def load_mnist():
    # Uso Keras apenas para baixar/carregar o MNIST, nao para treinar a rede.
    from keras.datasets import mnist

    return mnist.load_data()

def prepare_data(limit_train: int | None = None):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Achato cada imagem 28x28 para 784 entradas e normalizo os pixels.
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

    if limit_train is not None:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]

    # As labels viram one-hot porque a saida da rede tem 10 probabilidades.
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)
    return x_train, y_train, x_test, y_test

def plot_history(histories, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in histories.items():
        epochs = np.arange(1, len(history.train_loss) + 1)
        axes[0].plot(epochs, history.train_loss, marker="o", label=f"{name} treino")
        axes[0].plot(epochs, history.val_loss, marker="s", label=f"{name} teste")
        axes[1].plot(epochs, history.train_accuracy, marker="o", label=f"{name} treino")
        axes[1].plot(epochs, history.val_accuracy, marker="s", label=f"{name} teste")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_results(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "config",
                "layers",
                "activation",
                "learning_rate",
                "epochs",
                "batch_size",
                "test_loss",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    x_train, y_train, x_test, y_test = prepare_data(args.limit_train)

    experiments = {
        # Comparo duas arquiteturas para cumprir a exigencia da atividade.
        "base": {
            "layers": [784, 128, 64, 10],
            "activation": "relu",
            "learning_rate": 0.10,
        },
        "maior": {
            "layers": [784, 256, 128, 10],
            "activation": "relu",
            "learning_rate": 0.08,
        },
    }

    histories = {}
    rows = []

    for name, config in experiments.items():
        print(f"\n=== Experimento: {name} ===")
        model = MLP(
            layer_sizes=config["layers"],
            activation=config["activation"],
            learning_rate=config["learning_rate"],
            seed=args.seed,
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            x_val=x_test,
            y_val=y_test,
        )
        test_loss, test_acc = model.evaluate(x_test, y_test)
        histories[name] = history
        rows.append(
            {
                "config": name,
                "layers": "-".join(map(str, config["layers"])),
                "activation": config["activation"],
                "learning_rate": config["learning_rate"],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_loss": f"{test_loss:.6f}",
                "test_accuracy": f"{test_acc:.6f}",
            }
        )

    save_results(rows, results_dir / "experimentos.csv")
    plot_history(histories, results_dir / "curvas_treino.png")
    print("\nResultados salvos em results/experimentos.csv e results/curvas_treino.png")

if __name__ == "__main__":
    main()
