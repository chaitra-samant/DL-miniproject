"""Accident-anticipation models. All output frame-level logits (B, T, 2).

FrameRNN   baseline 1 (DSA-RNN style frame branch): frame embedding -> LSTM -> risk.
RiskNet    one class for baseline 2, baseline 3, the main model and the ablations:
  object encoder  MLP over the chosen object feature groups (app / geo / seg)
  interaction     "none":  objects are independent (baseline 2 is DSA-RNN style: no interaction)
                  "graph": one spatial graph-attention layer over the objects of a frame; attention logits
                           get a bias of -dist / tau (centre distance), the softmax-over-negative-distance
                           edge weighting used by STAGNet / Graph(Graph) spatial edges
  pooling         attention over objects conditioned on the previous hidden state (DSA-RNN dynamic spatial
                  attention); these weights are what the demo shows as "which object drives the risk"
  temporal        LSTM (or GRU) cell over [pooled objects, frame embedding] -> frame logits
Baseline 2 = RiskNet(app+geo, interaction none); baseline 3 = RiskNet(app+geo, graph);
main = RiskNet(app+geo+seg, graph).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = dict(app=384, geo=16, seg=9, frame=384)


class FrameRNN(nn.Module):
    def __init__(self, hidden=256, rnn="lstm", dropout=0.3, **_):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(DIMS["frame"], hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = (nn.LSTM if rnn == "lstm" else nn.GRU)(hidden, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, b):
        h, _ = self.rnn(self.enc(b["frame"]))
        return dict(logits=self.out(h), attn=None)


class RiskNet(nn.Module):
    def __init__(self, obj_groups=("app", "geo", "seg"), interaction="graph", use_frame=True, rnn="lstm",
                 hidden=256, heads=4, tau=0.2, dropout=0.3):
        super().__init__()
        self.groups, self.interaction, self.use_frame, self.rnn_type, self.tau = list(obj_groups), interaction, use_frame, rnn, tau
        self.enc = nn.ModuleDict({g: nn.Sequential(nn.Linear(DIMS[g], hidden), nn.ReLU()) for g in self.groups})
        self.obj_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * len(self.groups), hidden), nn.ReLU())
        if interaction == "graph":
            self.heads = heads
            self.qkv = nn.Linear(hidden, 3 * hidden)
            self.gat_out = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
        if use_frame:
            self.frame_enc = nn.Sequential(nn.Linear(DIMS["frame"], hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_h = nn.Linear(hidden, hidden)
        self.att_o = nn.Linear(hidden, hidden)
        self.att_w = nn.Linear(hidden, 1)
        cell_in = hidden * (2 if use_frame else 1)
        self.cell = nn.LSTMCell(cell_in, hidden) if rnn == "lstm" else nn.GRUCell(cell_in, hidden)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.hidden = hidden

    def interact(self, x, mask, centers):
        """x (B*T, N, H); one graph-attention layer with a distance bias, masked to real detections."""
        BT, N, H = x.shape
        q, k, v = self.qkv(x).view(BT, N, 3, self.heads, H // self.heads).unbind(2)
        logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / (H // self.heads) ** 0.5
        dist = torch.cdist(centers, centers)                                   # (BT, N, N), image-normalised
        logits = logits - (dist / self.tau).unsqueeze(1)
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)
        a = torch.softmax(logits, -1)
        y = torch.einsum("bhnm,bmhd->bnhd", a, v).reshape(BT, N, H)
        return self.norm(x + self.gat_out(y))

    def forward(self, b):
        B, T, N = b["mask"].shape
        mask = b["mask"]
        x = torch.cat([self.enc[g](b[g]) for g in self.groups], -1)            # (B, T, N, H*G)
        x = self.obj_proj(x)
        if self.interaction == "graph":
            geo = b["geo_raw"]                                                  # unnormalised geometry for distances
            centers = torch.stack([(geo[..., 6] + geo[..., 8]) / 2, (geo[..., 7] + geo[..., 9]) / 2], -1)
            x = self.interact(x.reshape(B * T, N, -1), mask.reshape(B * T, N), centers.reshape(B * T, N, 2)).view(B, T, N, -1)
        f = self.frame_enc(b["frame"]) if self.use_frame else None
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        o_proj = self.att_o(x)                                                  # (B, T, N, H)
        logits, attns = [], []
        any_obj = mask.any(-1)                                                  # (B, T)
        for t in range(T):
            e = self.att_w(torch.tanh(self.att_h(h)[:, None] + o_proj[:, t])).squeeze(-1)   # (B, N)
            e = e.masked_fill(~mask[:, t], -1e4)
            a = torch.softmax(e, -1) * any_obj[:, t, None]
            pooled = (a.unsqueeze(-1) * x[:, t]).sum(1)
            inp = torch.cat([pooled, f[:, t]], -1) if self.use_frame else pooled
            if self.rnn_type == "lstm":
                h, c = self.cell(inp, (h, c))
            else:
                h = self.cell(inp, h)
            logits.append(self.out(h))
            attns.append(a)
        return dict(logits=torch.stack(logits, 1), attn=torch.stack(attns, 1))


def build(spec):
    if spec["model"] == "frame":
        return FrameRNN(rnn=spec.get("rnn", "lstm"))
    return RiskNet(obj_groups=spec.get("groups", ("app", "geo", "seg")), interaction=spec.get("interaction", "graph"),
                   use_frame=spec.get("use_frame", True), rnn=spec.get("rnn", "lstm"))


def exp_loss(logits, y, toa, fps=10.0):
    """Exponential anticipation loss (Chan et al. 2016; UString _exp_loss). logits (B, T, 2), y (B,), toa (B,).
    Positive videos: frame t weighted by exp(-max(0, (toa - t - 1) / fps)); negative videos: plain CE."""
    B, T, _ = logits.shape
    t = torch.arange(T, device=logits.device)[None].float()
    ce_pos = F.cross_entropy(logits.reshape(-1, 2), torch.ones(B * T, dtype=torch.long, device=logits.device),
                             reduction="none").view(B, T)
    ce_neg = F.cross_entropy(logits.reshape(-1, 2), torch.zeros(B * T, dtype=torch.long, device=logits.device),
                             reduction="none").view(B, T)
    w = torch.exp(-torch.clamp((toa[:, None].float() - t - 1) / fps, min=0))
    yf = y[:, None].float()
    return (yf * w * ce_pos + (1 - yf) * ce_neg).mean()
