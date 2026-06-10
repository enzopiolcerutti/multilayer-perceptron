import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    # O clip evita log(0), que quebraria a loss.
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_true * np.log(clipped), axis=1)))

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Comparo a classe correta com a classe de maior probabilidade prevista.
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return float(np.mean(true_labels == pred_labels))

def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    # Transformo cada label em um vetor com 1 na classe correta.
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return encoded
