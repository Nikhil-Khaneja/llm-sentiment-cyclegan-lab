"""Nikhil's three sentiment models. All embeddings are randomly initialised and learned.

Each model maps (token_ids [B, T], lengths [B]) -> one logit per review (positive class).
"""
import torch
import torch.nn as nn


def _mask(x):
    return (x != 0).unsqueeze(-1).float()  # [B, T, 1], 0 at <pad>


class MeanPoolClassifier(nn.Module):
    """Baseline: embedding -> masked mean over tokens -> linear. A learned bag of words."""

    def __init__(self, vocab_size, emb_dim, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x, lengths):
        m = _mask(x)
        pooled = (self.emb(x) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled)).squeeze(-1)


class MultiKernelCNN(nn.Module):
    """Embedding -> parallel 1D convolutions (kernel 3/4/5 = tri/4/5-gram detectors)
    -> ReLU -> global max pool per filter -> concat -> linear."""

    def __init__(self, vocab_size, emb_dim, dropout, channels, kernel_sizes, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, x, lengths):
        e = self.emb(x).transpose(1, 2)                     # [B, E, T]
        pad = (x == 0).unsqueeze(1)                         # [B, 1, T]
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(e))[..., : x.size(1)]       # even kernels add one extra position
            h = h.masked_fill(pad, 0.0)                     # post-ReLU h >= 0, so padding can't win the max
            feats.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(-1)


class BiGRUClassifier(nn.Module):
    """Embedding -> 1-layer bidirectional GRU -> [final fwd state ; final bwd state] -> linear.

    Sequences are packed so the forward state is read at the true last token, not at padding.
    """

    def __init__(self, vocab_size, emb_dim, dropout, hidden, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                              # h: [2, B, H] (fwd, bwd)
        pooled = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(pooled)).squeeze(-1)


class LastTokenClassifier(nn.Module):
    """Member B baseline: embedding -> embedding of the last real token -> linear.

    Reviews are right-padded, so the last real token sits at index length-1. There is no context
    mixing at all, so this is a deliberately weak baseline (it only sees the review's final word).
    """

    def __init__(self, vocab_size, emb_dim, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x, lengths):
        last = x.gather(1, (lengths.to(x.device) - 1).unsqueeze(1).long()).squeeze(1)  # [B]
        return self.fc(self.drop(self.emb(last))).squeeze(-1)


class BiLSTMAttnClassifier(nn.Module):
    """Member B experiment: embedding -> 1-layer BiLSTM -> learned attention pooling -> linear.

    Attention: score_t = v . tanh(W h_t), softmax over real tokens only (padding masked to -inf),
    pooled = sum_t alpha_t h_t.
    """

    def __init__(self, vocab_size, emb_dim, dropout, hidden, attn_dim=None, **_):
        super().__init__()
        attn_dim = attn_dim or 2 * hidden
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(2 * hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # [B, T, 2H]
        scores = self.attn_v(torch.tanh(self.attn_proj(h))).squeeze(-1)                          # [B, T]
        # mask by length, not by pad id: a review that cleans to zero tokens has length 1 (one <pad>),
        # and masking on x == 0 would leave no real position -> softmax over all -inf -> NaN
        pad = torch.arange(x.size(1), device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        scores = scores.masked_fill(pad, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)                                       # [B, T, 1]
        pooled = (alpha * h).sum(1)
        return self.fc(self.drop(pooled)).squeeze(-1)


REGISTRY = {"mean_pool": MeanPoolClassifier, "cnn": MultiKernelCNN, "bigru": BiGRUClassifier,
            "last_token": LastTokenClassifier, "bilstm_attn": BiLSTMAttnClassifier}


def build(spec, vocab_size):
    return REGISTRY[spec["arch"]](vocab_size=vocab_size, **spec["params"])
