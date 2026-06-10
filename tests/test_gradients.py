import numpy as np
from mlp import MLP
from mlp.losses import cross_entropy, one_hot

def test_softmax_cross_entropy_gradient_check():
    # Crio um problema pequeno para testar se o backprop esta correto.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4))
    labels = np.array([0, 1, 2, 1, 0])
    y = one_hot(labels, 3)
    model = MLP([4, 6, 5, 3], learning_rate=0.01, seed=3)

    y_pred, cache = model.forward(x)
    gradients = model.backward(y, cache)

    # Perturbo alguns parametros e comparo o gradiente numerico com o analitico.
    epsilon = 1e-5
    checks = [
        ("weights", "dW", 0, (1, 2)),
        ("biases", "db", 0, (0, 3)),
        ("weights", "dW", 1, (2, 4)),
        ("weights", "dW", 2, (3, 1)),
        ("biases", "db", 2, (0, 2)),
    ]
    for parameter_name, gradient_name, layer_idx, position in checks:
        parameter = model.parameters[parameter_name][layer_idx]
        original_value = parameter[position]

        parameter[position] = original_value + epsilon
        loss_plus = cross_entropy(y, model.forward(x)[0])

        parameter[position] = original_value - epsilon
        loss_minus = cross_entropy(y, model.forward(x)[0])

        parameter[position] = original_value
        numerical_gradient = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytical_gradient = gradients[gradient_name][layer_idx][position]

        # Se os valores forem quase iguais, o gradiente daquela parte esta certo.
        assert np.isclose(numerical_gradient, analytical_gradient, atol=1e-6)
