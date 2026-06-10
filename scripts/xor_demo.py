import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp import MLP
from mlp.losses import one_hot

def main():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = one_hot(np.array([0, 1, 1, 0]), 2)

    model = MLP([2, 8, 8, 2], learning_rate=0.2, seed=10)
    model.fit(x, y, epochs=2000, batch_size=4, x_val=x, y_val=y, verbose=False)

    print("Predicoes XOR:", model.predict(x).tolist())
    print("Esperado:      ", [0, 1, 1, 0])

if __name__ == "__main__":
    main()
