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


def parse_words(words: list[str], n: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for w in words:
        if len(w) < n:
            continue
        chs = ["."] + list(w) + ["."]
        for i in range(len(chs) - n + 1):
            context = chs[i : i + n - 1]
            target = chs[i + n - 1]
            xs.append([stoi[ch] for ch in context])
            ys.append(stoi[target])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys


class NGram:
    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n
        self.weights = torch.randn((CHARS * (n - 1), CHARS), requires_grad=True)

    def train(
        self,
        words: list[str],
        epochs: int = 100,
        regularization: float = 0.01,
        learning_rate: float = 50,
        debug: bool = True,
    ):
        xs, ys = parse_words(words, self.n)

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
        xs, ys = parse_words(words, self.n)

        with torch.no_grad():
            xenc = F.one_hot(xs, num_classes=CHARS).float()
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
            loss = F.cross_entropy(logits, ys)

        return loss.item()

    def forward(self, x: str = "") -> str:
        context = [0]

        for c in x:
            if len(context) == self.n - 1:
                break
            if c not in stoi:
                raise ValueError(f"Character {c} not in stoi")
            context.append(stoi[c])

        while len(context) < self.n - 1:
            context.append(random.randint(1, 26))

        word = ""
        for c in context[1:]:
            word += itos[c]

        while True:
            # Forward pass
            # Neural net input: one-hot encoding
            xenc = F.one_hot(torch.tensor(context), num_classes=CHARS).float()
            # Predict log-counts
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
            counts = logits.exp()
            # Probabilities for next character
            p = counts / counts.sum(1, keepdim=True)

            # Sample from the distribution
            context.pop(0)
            context.append(torch.multinomial(p, num_samples=1, replacement=True).item())
            if context[-1] == 0:
                break

            word += itos[context[-1]]

        return word
