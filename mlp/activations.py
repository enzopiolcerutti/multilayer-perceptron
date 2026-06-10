import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    # Uso ReLU nas camadas ocultas para criar nao linearidade.
    return np.maximum(0.0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    # A derivada da ReLU vale 1 quando z > 0 e 0 no restante.
    return (z > 0.0).astype(z.dtype)

def sigmoid(z: np.ndarray) -> np.ndarray:
    # O clip evita overflow quando o valor de z e muito grande.
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t

def softmax(logits: np.ndarray) -> np.ndarray:
    # Subtraio o maximo para deixar o softmax numericamente estavel.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
}

def get_activation(name: str):
    try:
        return ACTIVATIONS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Ativacao desconhecida: {name}. Use uma de: {valid}") from exc
