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

        training_losses = []
        for i in range(epochs):
            # Shuffle the data
            permutation = torch.randperm(n)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            for k in range(0, n, minibatch_size):
                # Create the minibatches
                x_batch = x_shuffled[k : k + minibatch_size]
                y_batch = y_shuffled[k : k + minibatch_size]

                # Forward pass
                emb = self.embeddings[
                    x_batch
                ]  # (minibatch_size, block_size, embedding_dim)
                logits = torch.tanh(
                    emb.view(-1, self.inputs) @ self.weights[0] + self.biases[0]
                )  # (minibatch_size, weights[0].shape[1])
                for w, b in zip(self.weights[1:], self.biases[1:]):
                    logits = logits @ w + b  # (minibatch_size, w.shape[1])
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

            training_losses.append(loss.item())
            if debug:
                print(f"Epoch {i} Loss: {loss.item()}")

            # Reduce the learning rate on each quarter
            if i > 0 and i % (epochs // 4) == 0:
                learning_rate /= 10
                print(f"Learning rate reduced to {learning_rate}")

    def evaluate(self, words: list[str]) -> float:
        x, y = self.parse_words(words)

        with torch.no_grad():
            emb = self.embeddings[x]  # (x.shape[0], block_size, embedding_dim)
            logits = torch.tanh(
                emb.view(-1, self.inputs) @ self.weights[0] + self.biases[0]
            )  # (x.shape[0], weights[0].shape[1])
            for w, b in zip(self.weights[1:], self.biases[1:]):
                logits = logits @ w + b  # (x.shape[0], w.shape[1])
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
                logits = torch.tanh(
                    emb.view(1, -1) @ self.weights[0] + self.biases[0]
                )  # (1, weights[0].shape[1])
                for w, b in zip(self.weights[1:], self.biases[1:]):
                    logits = logits @ w + b  # (1, w.shape[1])
                probs = F.softmax(logits, dim=1)  # (1, CHARS)

                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                word += itos[ix]

        return word
