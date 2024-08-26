import random
import torch
import torch.nn.functional as F

chars: list[str] = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
stoi: dict[str, int] = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos: dict[int, str] = {i: s for s, i in stoi.items()}

CHARS = len(stoi)  # 27


def load_words(filename: str = "names.txt") -> tuple[list[str], list[str], list[str]]:
    words: list[str] = open(filename, "r").read().splitlines()

    random.shuffle(words)

    total_words = len(words)
    train_split = int(0.8 * total_words)
    dev_split = int(0.9 * total_words)

    train_words = words[:train_split]
    dev_words = words[train_split:dev_split]
    test_words = words[dev_split:]

    return train_words, dev_words, test_words


class NGram:
    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n
        self.weights = torch.randn((CHARS * (n - 1), CHARS), requires_grad=True)

    def parse_words(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for w in words:
            if len(w) < self.n:
                continue
            chs = ["."] + list(w) + ["."]
            for i in range(len(chs) - self.n + 1):
                context = chs[i : i + self.n - 1]
                target = chs[i + self.n - 1]
                xs.append([stoi[ch] for ch in context])
                ys.append(stoi[target])

        return torch.tensor(xs), torch.tensor(ys)

    def train(
        self,
        words: list[str],
        epochs: int = 100,
        regularization: float = 0.01,
        learning_rate: float = 50,
        debug: bool = True,
    ):
        xs, ys = self.parse_words(words)

        for i in range(epochs):
            # Forward pass
            # Neural net input: one-hot encoding
            xenc = F.one_hot(xs, num_classes=CHARS).float()
            # Predict log-counts
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights

            # Negative log likelihood + regularization
            loss = (
                F.cross_entropy(logits, ys) + regularization * (self.weights**2).mean()
            )

            if debug:
                print(f"Epoch {i} Loss: {loss.item()}")

            # Backward pass
            self.weights.grad = None
            loss.backward()

            # Update
            self.weights.data += -learning_rate * self.weights.grad

    def evaluate(self, words: list[str]) -> float:
        xs, ys = self.parse_words(words)

        with torch.no_grad():
            xenc = F.one_hot(xs, num_classes=CHARS).float()
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
            loss = F.cross_entropy(logits, ys)

        return loss.item()

    def forward(self, x: str = "") -> str:
        context = []
        for c in x:
            if len(context) == self.n - 1:
                break
            if c == ".":
                raise ValueError("Context cannot contain '.'")
            context.append(stoi[c])

        if len(context) == 0:
            context = [0]
        while len(context) < self.n - 1:
            context.append(random.randint(1, 26))

        word = ""
        for c in context:
            if c != 0:
                word += itos[c]

        with torch.no_grad():
            while True:
                # Forward pass
                # Neural net input: one-hot encoding
                xenc = F.one_hot(torch.tensor(context), num_classes=CHARS).float()
                # Predict log-counts
                logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
                # Probabilities for next character
                p = F.softmax(logits, dim=-1)

                # Sample from the distribution
                ix = torch.multinomial(p, num_samples=1, replacement=True).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                word += itos[ix]

        return word


class MLP:
    def __init__(
        self,
        sizes: list[int],
        block_size: int = 3,
        embedding_dim: int = 10,
        weight_scale: float = 0.01,
    ):
        self.block_size = block_size
        self.inputs = block_size * embedding_dim

        # C
        self.embeddings = torch.randn((CHARS, embedding_dim)) * weight_scale

        # Hidden layers
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []
        for i in range(len(sizes)):
            inputs = self.inputs if i == 0 else sizes[i - 1]
            neurons = sizes[i]

            self.weights.append(torch.randn((inputs, neurons)) * weight_scale)
            self.biases.append(torch.randn((neurons)))

        # Output layer
        self.weights.append(torch.randn((sizes[-1], CHARS)) * weight_scale)
        self.biases.append(torch.randn((CHARS)))

        self.parameters = [self.embeddings] + self.weights + self.biases
        for p in self.parameters:
            p.requires_grad = True

    def parse_words(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []
        for w in words:
            context = [0] * self.block_size
            for ch in w + ".":
                ix = stoi[ch]
                x.append(context)
                y.append(ix)
                context = context[1:] + [ix]

        return torch.tensor(x), torch.tensor(y)

    def train(
        self,
        words: list[str],
        minibatch_size: int = 32,
        epochs: int = 40,
        regularization: float = 0.01,
        learning_rate: float = 0.1,
        debug: bool = True,
    ):
        x, y = self.parse_words(words)
        n = x.shape[0]

        training_losses: list[float] = []
        for i in range(epochs):
            # Shuffle the data
            permutation = torch.randperm(n)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            epoch_losses: list[float] = []
            for k in range(0, n, minibatch_size):
                # Create the minibatches
                x_batch = x_shuffled[k : k + minibatch_size]
                y_batch = y_shuffled[k : k + minibatch_size]

                # Forward pass
                emb = self.embeddings[
                    x_batch
                ]  # (minibatch_size, block_size, embedding_dim)
                hidden = torch.tanh(
                    emb.view(-1, self.inputs) @ self.weights[0] + self.biases[0]
                )  # (minibatch_size, weights[0].shape[1])
                for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
                    hidden = torch.tanh(hidden @ w + b)  # (minibatch_size, w.shape[1])
                logits = (
                    hidden @ self.weights[-1] + self.biases[-1]
                )  # (minibatch_size, CHARS)
                loss = F.cross_entropy(logits, y_batch)
                # Regularization
                for w in self.weights:
                    loss += regularization * (w**2).mean()

                # Backward pass
                for p in self.parameters:
                    p.grad = None
                loss.backward()

                # Update
                for p in self.parameters:
                    p.data += -learning_rate * p.grad

                epoch_losses.append(loss.item())

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(epoch_loss)
            if debug:
                print(f"Epoch {i} Loss: {epoch_loss}")

            # Reduce the learning rate after half
            if i > 0 and i % (epochs // 2) == 0:
                learning_rate /= 10
                print(f"Learning rate reduced to {learning_rate}")

    def evaluate(self, words: list[str]) -> float:
        x, y = self.parse_words(words)

        with torch.no_grad():
            emb = self.embeddings[x]  # (x.shape[0], block_size, embedding_dim)
            hidden = torch.tanh(
                emb.view(-1, self.inputs) @ self.weights[0] + self.biases[0]
            )  # (x.shape[0], weights[0].shape[1])
            for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
                hidden = torch.tanh(hidden @ w + b)  # (x.shape[0], w.shape[1])
            logits = hidden @ self.weights[-1] + self.biases[-1]  # (x.shape[0], CHARS)
            loss = F.cross_entropy(logits, y)

        return loss.item()

    def forward(self, x: str = "") -> str:
        context = [0] * self.block_size
        for c in x:
            ix = stoi[c]
            context = context[1:] + [ix]

        word = ""
        for c in context:
            if c != 0:
                word += itos[c]

        with torch.no_grad():
            while True:
                emb = self.embeddings[
                    torch.tensor([context])
                ]  # (1, block_size, embedding_dim)
                hidden = torch.tanh(
                    emb.view(1, -1) @ self.weights[0] + self.biases[0]
                )  # (1, weights[0].shape[1])
                for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
                    hidden = torch.tanh(hidden @ w + b)  # (1, w.shape[1])
                logits = hidden @ self.weights[-1] + self.biases[-1]  # (1, CHARS)
                probs = F.softmax(logits, dim=1)  # (1, CHARS)

                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                word += itos[ix]

        return word


class BatchNorm:
    def __init__(
        self,
        sizes: list[int],
        block_size: int = 3,
        embedding_dim: int = 10,
        weight_scale: float = 0.01,
        eps: float = 1.0e-5,
    ):
        self.block_size = block_size
        self.inputs = block_size * embedding_dim
        self.eps = eps

        # C
        self.embeddings = torch.randn((CHARS, embedding_dim))

        # Hidden layers
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor | None] = []
        self.gamma: list[torch.Tensor] = []
        self.beta: list[torch.Tensor] = []
        self.mean: list[torch.Tensor] = []
        self.std: list[torch.Tensor] = []
        for i in range(len(sizes)):
            inputs = self.inputs if i == 0 else sizes[i - 1]
            neurons = sizes[i]

            self.weights.append(
                torch.randn((inputs, neurons)) * ((5 / 3) / (inputs**0.5))
            )
            self.biases.append(None)
            self.gamma.append(torch.ones((1, neurons)))
            self.beta.append(torch.zeros((1, neurons)))
            self.mean.append(torch.zeros((1, neurons)))
            self.std.append(torch.zeros((1, neurons)))

        # Output layer
        self.weights.append(torch.randn((sizes[-1], CHARS)) * weight_scale)
        self.biases.append(torch.randn((CHARS)) * weight_scale)

        self.parameters = (
            [self.embeddings]
            + self.weights
            + [bias for bias in self.biases if bias is not None]
            + self.gamma
            + self.beta
        )
        for p in self.parameters:
            p.requires_grad = True

    def parse_words(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []
        for w in words:
            context = [0] * self.block_size
            for ch in w + ".":
                ix = stoi[ch]
                x.append(context)
                y.append(ix)
                context = context[1:] + [ix]

        return torch.tensor(x), torch.tensor(y)

    def train(
        self,
        words: list[str],
        minibatch_size: int = 32,
        epochs: int = 40,
        momentum: float = 0.001,
        learning_rate: float = 0.1,
        debug: bool = True,
    ):
        x, y = self.parse_words(words)
        n = x.shape[0]

        training_losses: list[float] = []
        for i in range(epochs):
            # Shuffle the data
            permutation = torch.randperm(n)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            epoch_losses: list[float] = []
            for k in range(0, n, minibatch_size):
                # Create the minibatches
                x_batch = x_shuffled[k : k + minibatch_size]
                y_batch = y_shuffled[k : k + minibatch_size]

                # Forward pass
                emb = self.embeddings[
                    x_batch
                ]  # (minibatch_size, block_size, embedding_dim)
                hpreact = emb.view(
                    emb.shape[0], -1
                )  # embcat = (minibatch_size, block_size * embedding_dim)

                # Hidden layers
                for layer, (w, b, gamma, beta) in enumerate(
                    zip(self.weights[:-1], self.biases[:-1], self.gamma, self.beta)
                ):
                    # Linear layer
                    hpreact = hpreact @ w  # (minibatch_size, w.shape[1])
                    if b is not None:
                        hpreact += b  # (minibatch_size, b.shape[0])

                    # Batch normalization
                    bnmean = hpreact.mean(0, keepdim=True)  # (1, w.shape[1])
                    bnstd = hpreact.std(0, keepdim=True)  # (1, w.shape[1])
                    hpreact = (
                        gamma * (hpreact - bnmean) / torch.sqrt(bnstd + self.eps) + beta
                    )  # (minibatch_size, w.shape[1])

                    # Keep track of the running batchnorm mean and std for inference
                    with torch.no_grad():
                        self.mean[layer] = (1 - momentum) * self.mean[
                            layer
                        ] + momentum * bnmean  # (1, w.shape[1])
                        self.std[layer] = (1 - momentum) * self.std[
                            layer
                        ] + momentum * bnstd  # (1, w.shape[1])

                    # Activation function
                    hpreact = torch.tanh(hpreact)  # (minibatch_size, w.shape[1])

                # Output layer
                logits = hpreact @ self.weights[-1]  # (minibatch_size, CHARS)
                if self.biases[-1] is not None:
                    logits += self.biases[-1]  # (minibatch_size, CHARS)
                loss = F.cross_entropy(logits, y_batch)

                # Backward pass
                for p in self.parameters:
                    p.grad = None
                loss.backward()

                # Update
                for p in self.parameters:
                    p.data += -learning_rate * p.grad

                epoch_losses.append(loss.item())

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(epoch_loss)
            if debug:
                print(f"Epoch {i} Loss: {epoch_loss}")

            # Reduce the learning rate after half
            if i > 0 and i % (epochs // 2) == 0:
                learning_rate /= 10
                print(f"Learning rate reduced to {learning_rate}")

    def evaluate(self, words: list[str]) -> float:
        x, y = self.parse_words(words)

        with torch.no_grad():
            emb = self.embeddings[x]  # (x.shape[0], block_size, embedding_dim)
            hpreact = emb.view(
                emb.shape[0], -1
            )  # embcat = (x.shape[0], block_size * embedding_dim)

            # Hidden layers
            for layer, (w, b, gamma, beta) in enumerate(
                zip(self.weights[:-1], self.biases[:-1], self.gamma, self.beta)
            ):
                # Linear layer
                hpreact = hpreact @ w  # (x.shape[0], w.shape[1])
                if b is not None:
                    hpreact += b  # (x.shape[0], b.shape[0])

                # Batch normalization
                hpreact = (
                    gamma
                    * (hpreact - self.mean[layer])
                    / torch.sqrt(self.std[layer] + self.eps)
                    + beta
                )  # (x.shape[0], w.shape[1])

                # Activation function
                hpreact = torch.tanh(hpreact)  # (x.shape[0], w.shape[1])

            # Output layer
            logits = hpreact @ self.weights[-1]  # (x.shape[0], CHARS)
            if self.biases[-1] is not None:
                logits += self.biases[-1]  # (x.shape[0], CHARS)
            loss = F.cross_entropy(logits, y)

        return loss.item()

    def forward(self, x: str = "") -> str:
        context = [0] * self.block_size
        for c in x:
            ix = stoi[c]
            context = context[1:] + [ix]

        word = ""
        for c in context:
            if c != 0:
                word += itos[c]

        with torch.no_grad():
            while True:
                emb = self.embeddings[
                    torch.tensor([context])
                ]  # (1, block_size, embedding_dim)
                hpreact = emb.view(
                    emb.shape[0], -1
                )  # embcat = (1, block_size * embedding_dim)

                # Hidden layers
                for layer, (w, b, gamma, beta) in enumerate(
                    zip(self.weights[:-1], self.biases[:-1], self.gamma, self.beta)
                ):
                    # Linear layer
                    hpreact = hpreact @ w  # (1, w.shape[1])
                    if b is not None:
                        hpreact += b  # (1, b.shape[0])

                    # Batch normalization
                    hpreact = (
                        gamma
                        * (hpreact - self.mean[layer])
                        / torch.sqrt(self.std[layer] + self.eps)
                        + beta
                    )  # (1, w.shape[1])

                    # Activation function
                    hpreact = torch.tanh(hpreact)  # (1, w.shape[1])

                # Output layer
                logits = hpreact @ self.weights[-1]  # (1, CHARS)
                if self.biases[-1] is not None:
                    logits += self.biases[-1]  # (1, CHARS)
                probs = F.softmax(logits, dim=1)  # (1, CHARS)

                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                word += itos[ix]

        return word


class PytorchifiedModule:
    def __init__(self):
        self.training: bool = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def parameters(self) -> list[torch.Tensor]:
        pass


class Linear(PytorchifiedModule):
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        super().__init__()
        self.weights: torch.Tensor = torch.randn((fan_in, fan_out)) / (fan_in**0.5)
        self.biases: torch.Tensor = torch.zeros((fan_out)) if bias else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x @ self.weights
        if self.biases is not None:
            self.out += self.biases
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.weights] + ([self.biases] if self.biases is not None else [])


class BatchNorm1d(PytorchifiedModule):
    def __init__(self, dim: int, eps: float = 1.0e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # Parameters (trained with backprop)
        self.gamma: torch.Tensor = torch.ones(dim)
        self.beta: torch.Tensor = torch.zeros(dim)
        # Buffers (trained with a running momentum update)
        self.running_mean: torch.Tensor = torch.zeros(dim)
        self.running_var: torch.Tensor = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)  # Batch mean
            xvar = x.var(0, keepdim=True)  # Batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)

        self.out = self.gamma * xhat + self.beta

        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh(PytorchifiedModule):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        super().__init__()
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []


class PytorchifiedBatchNorm:
    def __init__(
        self,
        sizes: list[int] = [200, 200, 200, 200, 200],
        block_size: int = 3,
        embedding_dim: int = 10,
        gain: float = 5 / 3,
        weight_scale: float = 0.1,
        batchnorm: bool = True,
        activation: bool = True,
    ):
        self.block_size = block_size
        self.inputs = block_size * embedding_dim

        self.embeddings = torch.randn((CHARS, embedding_dim))
        self.layers: list[PytorchifiedModule] = []

        # Hidden layers
        for i in range(len(sizes)):
            inputs = self.inputs if i == 0 else sizes[i - 1]
            neurons = sizes[i]

            layer = Linear(inputs, neurons, bias=not batchnorm)
            with torch.no_grad():
                layer.weights *= gain
            self.layers.append(layer)

            if batchnorm:
                self.layers.append(BatchNorm1d(neurons))

            if activation:
                self.layers.append(Tanh())

        # Output layer
        layer = Linear(sizes[-1], CHARS, bias=not batchnorm)
        with torch.no_grad():
            layer.weights *= weight_scale
        self.layers.append(layer)

        if batchnorm:
            self.layers.append(BatchNorm1d(CHARS))

        self.parameters = [self.embeddings] + [
            p for layer in self.layers for p in layer.parameters()
        ]
        for p in self.parameters:
            p.requires_grad = True

    def parse_words(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []
        for w in words:
            context = [0] * self.block_size
            for ch in w + ".":
                ix = stoi[ch]
                x.append(context)
                y.append(ix)
                context = context[1:] + [ix]

        return torch.tensor(x), torch.tensor(y)

    def train(
        self,
        words: list[str],
        minibatch_size: int = 32,
        epochs: int = 40,
        learning_rate: float = 0.1,
        debug: bool = True,
    ) -> list[list[float]]:
        x, y = self.parse_words(words)
        n = x.shape[0]
        ud: list[list[float]] = []

        for layer in self.layers:
            layer.training = True

        training_losses: list[float] = []
        for i in range(epochs):
            # Shuffle the data
            permutation = torch.randperm(n)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            epoch_losses: list[float] = []
            for k in range(0, n, minibatch_size):
                # Create the minibatches
                x_batch = x_shuffled[k : k + minibatch_size]
                y_batch = y_shuffled[k : k + minibatch_size]

                # Forward pass
                emb = self.embeddings[
                    x_batch
                ]  # (minibatch_size, block_size, embedding_dim)
                x_out = emb.view(
                    emb.shape[0], -1
                )  # (minibatch_size, block_size * embedding_dim)
                for layer in self.layers:
                    x_out = layer(x_out)
                loss = F.cross_entropy(x_out, y_batch)

                # Backward pass
                if debug:
                    for layer in self.layers:
                        layer.out.retain_grad()
                for p in self.parameters:
                    p.grad = None
                loss.backward()

                # Update
                for p in self.parameters:
                    p.data += -learning_rate * p.grad

                epoch_losses.append(loss.item())
                with torch.no_grad():
                    ud.append(
                        [
                            ((learning_rate * p.grad).std() / p.data.std())
                            .log10()
                            .item()
                            for p in self.parameters
                        ]
                    )

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(epoch_loss)
            if debug:
                print(f"Epoch {i} Loss: {epoch_loss}")

            # Reduce the learning rate after half
            if i > 0 and i % (epochs // 2) == 0:
                learning_rate /= 10
                print(f"Learning rate reduced to {learning_rate}")

        return ud

    def evaluate(self, words: list[str], batchnorm: bool = True) -> float:
        x, y = self.parse_words(words)

        for layer in self.layers:
            layer.training = False

        with torch.no_grad():
            # Forward pass
            emb = self.embeddings[x]  # (x.shape[0], block_size, embedding_dim)
            x_out = emb.view(
                emb.shape[0], -1
            )  # (x.shape[0], block_size * embedding_dim)

            for i, layer in enumerate(self.layers):
                if batchnorm:
                    x_out = layer(x_out)
                else:
                    if (
                        i + 1 < len(self.layers)
                        and isinstance(layer, Linear)
                        and isinstance(self.layers[i + 1], BatchNorm1d)
                    ):
                        bn_layer: BatchNorm1d = self.layers[i + 1]
                        folded_layer = Linear(
                            layer.weights.shape[0], layer.weights.shape[1], bias=True
                        )

                        folded_layer.weights = layer.weights * (
                            bn_layer.gamma
                            / torch.sqrt(bn_layer.running_var + bn_layer.eps)
                        )
                        if layer.biases is not None:
                            folded_layer.biases = (
                                layer.biases - bn_layer.running_mean
                            ) * (
                                bn_layer.gamma
                                / torch.sqrt(bn_layer.running_var + bn_layer.eps)
                            ) + bn_layer.beta
                        else:
                            folded_layer.biases = (
                                -bn_layer.running_mean
                                * (
                                    bn_layer.gamma
                                    / torch.sqrt(bn_layer.running_var + bn_layer.eps)
                                )
                                + bn_layer.beta
                            )

                        x_out = folded_layer(x_out)
                    elif isinstance(layer, Tanh):
                        x_out = layer(x_out)

            loss = F.cross_entropy(x_out, y)

        return loss.item()

    def forward(self, x: str = "") -> str:
        context = [0] * self.block_size
        for c in x:
            ix = stoi[c]
            context = context[1:] + [ix]

        word = ""
        for c in context:
            if c != 0:
                word += itos[c]

        for layer in self.layers:
            layer.training = False

        with torch.no_grad():
            while True:
                emb = self.embeddings[
                    torch.tensor([context])
                ]  # (1, block_size, embedding_dim)
                x_out = emb.view(emb.shape[0], -1)  # (1, block_size * embedding_dim)

                for layer in self.layers:
                    x_out = layer(x_out)

                probs = F.softmax(x_out, dim=1)  # (1, CHARS)

                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                word += itos[ix]

        return word
