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


class NGram:
    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be at least 2")

        self.n = n
        self.weights = torch.randn((CHARS * (n - 1), CHARS), requires_grad=True)
        self.counts: dict[tuple[str, ...], int] = {}

    def train(
        self,
        words: list[str],
        epochs: int = 100,
        regularization: float = 0.01,
        learning_rate: float = 50,
        debug: bool = True,
    ):
        # Parse words into sequences of characters
        xs, ys = [], []
        for w in words:
            if len(w) < self.n:
                continue
            chs = ["."] + list(w) + ["."]
            for i in range(len(chs) - self.n + 1):  # Changed this line
                context = chs[i : i + self.n - 1]
                target = chs[i + self.n - 1]
                xs.append([stoi[ch] for ch in context])
                ys.append(stoi[target])
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        for i in range(epochs):
            # Forward pass
            # Neural net input: one-hot encoding
            xenc = F.one_hot(xs, num_classes=CHARS).float()
            # Predict log-counts
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
            counts = logits.exp()
            # Probabilities for next character
            probs = counts / counts.sum(dim=1, keepdim=True)

            # Negative log likelihood
            loss = (
                -probs[torch.arange(ys.shape[0]), ys].log().mean()
                + regularization * (self.weights**2).mean()  # Regularization
            )

            if debug:
                print(f"Epoch {i} Loss: {loss.item()}")

            # Backward pass
            self.weights.grad = None
            loss.backward()

            # Update
            self.weights.data += -learning_rate * self.weights.grad

    def forward(self, x: str = "") -> str:
        ix = 0
        word = ""

        for c in x:
            if c not in stoi:
                raise ValueError(f"Character {c} not in stoi")

            ix = stoi[c]

        while True:
            # Forward pass
            # Neural net input: one-hot encoding
            xenc = F.one_hot(torch.tensor([ix]), num_classes=CHARS).float()
            # Predict log-counts
            logits = xenc.view(-1, CHARS * (self.n - 1)) @ self.weights
            counts = logits.exp()
            # Probabilities for next character
            p = counts / counts.sum(1, keepdim=True)

            # Sample from the distribution
            ix = torch.multinomial(p, num_samples=1, replacement=True).item()
            if ix == 0:
                break

            word += itos[ix]

        return word


if __name__ == "__main__":
    ngram = NGram(2)
    ngram.train(["emma"])
    print(ngram.forward())
