import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = inputs.shape

        # idx and targets are both (B,T) tensor of integers
        # (B, T, C)
        tok_emb = self.token_embedding_table(
            inputs
        )  # (idx.shape[0], idx.shape[1], vocab_size)
        # (T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)  # [0, 1, 2, ..., T-1]
        )
        # (B, T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # (idx.shape[0], idx.shape[1], vocab_size)

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


model = BigramLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd)
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
