import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    """Encoder: take a string, output a list of integers"""
    return [stoi[c] for c in s]


def decode(encoded: list[int]) -> str:
    """Decoder: take a list of integers, output a string"""
    return "".join([itos[i] for i in encoded])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, head_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Non-parameter
        self.tril: torch.Tensor
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch, time, channels
        B, T, C = x.shape
        k: torch.Tensor = self.key(x)  # (B, T, head_size = n_embd//n_head)
        q: torch.Tensor = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities")
        wei: torch.Tensor = (
            q @ k.transpose(-2, -1) * C**-0.5  # Scaled attention
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v: torch.Tensor = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads"""

    def __init__(self, num_heads: int, head_size: int, block_size: int, n_embd: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, block_size=block_size, n_embd=n_embd)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention output
        out = torch.cat(
            [h(x) for h in self.heads],  # [(B, T, head_size = n_embd//n_head)...]
            dim=-1,
        )  # (B, T, n_embd)
        out = self.dropout(self.proj(out))  # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """Feed-forward layer"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, C)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head, head_size=head_size, block_size=block_size, n_embd=n_embd
        )
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
    ):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Multi-head attention
        self.blocks = nn.Sequential(
            *[
                Block(n_embd=n_embd, n_head=n_head, block_size=block_size)
                for _ in range(n_layer)
            ],
        )
        # Layernorm
        self.ln_f = nn.LayerNorm(n_embd)
        # Decoder language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # batch, time
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)  # [0, 1, 2, ..., T-1]
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) -> (B, T, C)

        # Apply one head of self-attention
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # predict
            # (B, T, C) = batch, time, channels
            logits, loss = self(idx_cond)  # (idx.shape[0], i + 1, vocab_size)
            # focus only on the last time step
            # becomes (B, C)
            logits = logits[:, -1, :]  # (idx.shape[0], vocab_size)
            # apply softmax to get probabilities
            # (B, C)
            probs = F.softmax(logits, dim=-1)  # (idx.shape[0], vocab_size)
            # sample from the distribution
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (idx.shape[0], 1)
            # append sampled index to the running sequence
            # (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)  # (idx.shape[0], i + 2)

        return idx


model = BigramLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
