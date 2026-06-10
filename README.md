# MLP (Multi-Layer Perceptron)

Este repositório implementa um Multi-Layer Perceptron para classificar dígitos do MNIST sem PyTorch, TensorFlow ou frameworks de deep learning na parte matemática da rede. O NumPy é usado para operações matriciais, e `keras.datasets.mnist` é usado apenas para carregar os dados.

## Como rodar

Crie um ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Rode primeiro o exemplo XOR, que é um teste pequeno para validar se a rede consegue aprender um padrão não linear:

```bash
python scripts/xor_demo.py
```

Rode o teste de gradientes:

```bash
python -m pytest tests
```

Treine no MNIST:

```bash
python scripts/train_mnist.py --epochs 12 --batch-size 128
```

Para fazer um teste rápido sem treinar nos 60.000 exemplos:

```bash
python scripts/train_mnist.py --epochs 3 --batch-size 128 --limit-train 5000
```

Os resultados são salvos em:

- `results/experimentos.csv`: tabela comparativa das configurações.
- `results/curvas_treino.png`: curva de loss e acurácia por época.

## Arquitetura escolhida

A implementação aceita um número arbitrário de camadas por meio da lista `layer_sizes`. Por exemplo:

```python
MLP(layer_sizes=[784, 128, 64, 10], activation="relu", learning_rate=0.10)
```

Essa arquitetura significa:

- `784` entradas: cada imagem MNIST tem 28 x 28 pixels, achatados em um vetor.
- `128` neurônios na primeira camada oculta.
- `64` neurônios na segunda camada oculta.
- `10` neurônios na saída, um para cada dígito de 0 a 9.

Usei ReLU nas camadas ocultas porque ela reduz o problema de saturação que aparece com sigmoid e tanh. Em sigmoid, quando os valores ficam muito positivos ou muito negativos, a derivada fica muito próxima de zero, e o gradiente quase não passa para camadas anteriores. A ReLU mantém derivada 1 para entradas positivas, o que costuma acelerar o treinamento em redes densas.

Na saída, usei softmax. A softmax transforma os logits da última camada em uma distribuição de probabilidades:

```text
p_i = exp(z_i) / soma_j exp(z_j)
```

Como a tarefa é classificação multiclasse, a loss usada é cross-entropy:

```text
L = -media soma_i y_i log(p_i)
```

A combinação softmax + cross-entropy tem uma simplificação importante no backpropagation:

```text
dL/dlogits = (y_pred - y_true) / batch_size
```

Isso deixa o código mais simples e numericamente mais estável do que calcular a jacobiana completa da softmax.

## Como o forward funciona

Cada camada densa calcula:

```text
Z = A_anterior W + b
A = ativacao(Z)
```

Nas camadas ocultas, `A` passa pela ReLU. Na camada final, os logits passam pela softmax. Durante o forward, a rede guarda os valores de `Z` e `A` em um cache, porque o backward precisa deles para calcular os gradientes.

## Como o backpropagation funciona

O backpropagation aplica a regra da cadeia de trás para frente. Primeiro calculamos o erro da camada de saída:

```text
delta_saida = (y_pred - y_true) / batch_size
```

Depois, para cada camada:

```text
dW = A_anterior.T @ delta
db = soma(delta)
delta_anterior = (delta @ W.T) * derivada_ativacao(Z_anterior)
```

`dW` mede quanto cada peso contribuiu para aumentar ou diminuir a loss. `db` faz o mesmo para o viés. O SGD atualiza os parâmetros na direção oposta ao gradiente:

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

## Inicialização dos pesos

Para ReLU, usei inicialização He:
 
```text
W ~ Normal(0, sqrt(2 / fan_in))
```

Isso ajuda a manter a variância das ativações mais estável entre as camadas. Se os pesos começarem grandes demais, os logits podem explodir; se começarem pequenos demais, os sinais podem desaparecer. Inicializar todos os pesos com zero também é uma má ideia, porque todos os neurônios da mesma camada receberiam o mesmo gradiente e aprenderiam exatamente a mesma coisa.

## Resultados

O script `scripts/train_mnist.py` compara duas configurações:

| Configuração | Camadas | Ativação | Learning rate |
| --- | --- | --- | --- |
| `base` | 784-128-64-10 | ReLU | 0.10 |
| `maior` | 784-256-128-10 | ReLU | 0.08 |

Depois de executar o treino completo, os valores finais ficaram:

| Configuração | Loss teste | Acurácia teste |
| --- | ---: | ---: |
| `base` | 0.076473 | 97.57% |
| `maior` | 0.080685 | 97.43% |

A meta da atividade era atingir acurácia maior ou igual a 92% no teste. As duas configurações passaram da meta. A configuração `base` ficou ligeiramente melhor, mesmo sendo menor, o que sugere que a rede maior não trouxe ganho relevante para esse número de épocas e learning rate.

## Decisões e dificuldades

### Qual foi a decisão técnica mais difícil que você tomou? Por que fez essa escolha?

A decisão mais importante foi implementar o backward de forma vetorizada e usando a simplificação matemática de softmax com cross-entropy. Eu escolhi esse caminho porque ele deixa a implementação mais direta e diminui o risco de erro numérico. Em vez de calcular a derivada completa da softmax classe por classe, o erro da saída vira `y_pred - y_true`.

### O que você tentou que não funcionou? O que aprendeu com isso?

Um ponto que eu tive dificuldade foi com os pesos. Se os pesos forem inicializados com valores errados, a rede até vai executar, mas a loss pode cair muito devagar ou ficar instável. Por isso usei inicialização He quando a ativação é ReLU.

Outro ponto sensível é esquecer a divisão pelo `batch_size` no gradiente da saída. Sem essa divisão, o tamanho do processo passa a depender do tamanho do batch, e o learning rate fica mais difícil de interpretar.

### Se fosse refazer do zero, o que faria diferente?

Eu começaria validando tudo em um problema pequeno, como XOR, antes de treinar no MNIST. Também manteria o gradient check desde o início, porque ele ia me mostrar erros de sinal, transposição e derivada muito antes de aparecerem como uma loss fora do comum no treino completo.

## Estrutura

```text
.
├── README.md
├── mlp/
│   ├── __init__.py
│   ├── activations.py
│   ├── losses.py
│   ├── network.py
│   └── optimizers.py
├── notebooks/
├── results/
├── scripts/
│   ├── train_mnist.py
│   └── xor_demo.py
├── tests/
│   └── test_gradients.py
└── requirements.txt
```