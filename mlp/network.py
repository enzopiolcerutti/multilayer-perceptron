from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import get_activation, softmax
from .losses import accuracy, cross_entropy
from .optimizers import SGD


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]


class MLP:
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        seed: int | None = 42,
    ):
        if len(layer_sizes) < 3:
            raise ValueError("Use pelo menos entrada, uma camada oculta e saida.")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation(activation)
        self.optimizer = SGD(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self):
        weights = []
        biases = []

        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Para ReLU, a inicializacao He ajuda o sinal a nao explodir nem sumir.
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            weights.append(self.rng.normal(0.0, scale, size=(fan_in, fan_out)))
            biases.append(np.zeros((1, fan_out), dtype=np.float64))

        return {"weights": weights, "biases": biases}

    def forward(self, x: np.ndarray):
        # Guardo A e Z porque vou precisar deles no backpropagation.
        activations = [x]
        pre_activations = []
        a = x

        hidden_layers = len(self.parameters["weights"]) - 1
        for layer_idx in range(hidden_layers):
            # Cada camada oculta faz: soma ponderada e depois ativacao.
            z = a @ self.parameters["weights"][layer_idx] + self.parameters["biases"][layer_idx]
            a = self.activation(z)
            pre_activations.append(z)
            activations.append(a)

        # Na saida eu uso softmax para obter probabilidades das 10 classes.
        logits = a @ self.parameters["weights"][-1] + self.parameters["biases"][-1]
        y_pred = softmax(logits)
        pre_activations.append(logits)
        activations.append(y_pred)

        cache = {"A": activations, "Z": pre_activations}
        return y_pred, cache

    def backward(self, y_true: np.ndarray, cache):
        weights = self.parameters["weights"]
        activations = cache["A"]
        pre_activations = cache["Z"]
        batch_size = y_true.shape[0]

        dW = [None] * len(weights)
        db = [None] * len(weights)

        # Com softmax + cross-entropy, o gradiente da saida fica simples.
        delta = (activations[-1] - y_true) / batch_size

        for layer_idx in reversed(range(len(weights))):
            # Calculo quanto cada peso e vies contribuiu para o erro.
            dW[layer_idx] = activations[layer_idx].T @ delta
            db[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                # Propago o erro para a camada anterior usando a regra da cadeia.
                delta = (delta @ weights[layer_idx].T) * self.activation_derivative(
                    pre_activations[layer_idx - 1]
                )

        return {"dW": dW, "db": db}

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        y_pred, cache = self.forward(x_batch)
        loss = cross_entropy(y_batch, y_pred)
        gradients = self.backward(y_batch, cache)
        self.optimizer.update(self.parameters, gradients)
        return loss

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        history = TrainingHistory([], [], [], [])
        n_samples = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Em cada epoca eu embaralho os dados para variar os mini-batches.
            indices = self.rng.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            batch_losses = []
            for start in range(0, n_samples, batch_size):
                # Cada mini-batch faz forward, backward e atualizacao dos pesos.
                end = start + batch_size
                loss = self.train_batch(x_shuffled[start:end], y_shuffled[start:end])
                batch_losses.append(loss)

            train_loss, train_acc = self.evaluate(x_train, y_train)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)

            if x_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(x_val, y_val)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

            if verbose:
                print(
                    f"epoch={epoch:02d} "
                    f"batch_loss={np.mean(batch_losses):.4f} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_pred, _ = self.forward(x)
        return y_pred

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y_pred = self.predict_proba(x)
        return cross_entropy(y, y_pred), accuracy(y, y_pred)
