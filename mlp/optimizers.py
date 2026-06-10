class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        for idx in range(len(parameters["weights"])):
            # O SGD anda na direcao oposta ao gradiente para reduzir a loss.
            parameters["weights"][idx] -= self.learning_rate * gradients["dW"][idx]
            parameters["biases"][idx] -= self.learning_rate * gradients["db"][idx]
