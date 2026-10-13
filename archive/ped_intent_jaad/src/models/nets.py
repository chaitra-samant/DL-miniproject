"""Models. One network class covers the bbox-only LSTM, the PIEPredict-style model and the main model.

CueNet:
  per cue group g: MLP encoder e_g(t) = f_g(x_g(t))
  fusion: gated  -> a(t) = softmax(W [e_1..e_G]) over groups, z(t) = sum_g a_g(t) e_g(t)
                    (a(t) is reported as cue reliance over time)
          concat -> z(t) = W [e_1..e_G]
  encoder: LSTM or GRU over z(1..T)
  heads:   crossing logit from the last hidden state
           trajectory decoder: LSTMCell unrolled pred_len steps, K hypotheses via learned mode
           embeddings, plus mode logits; outputs box offsets from the last observed box
  dec_cond (PIEPredict-style): the decoder input at every step also gets the estimated crossing
           probability and the ego state of the last observed frame, as in PIEPredict's decoder
           (which uses intention probability and vehicle speed; JAAD has no speed, so ego action).
"""
import numpy as np
import torch
import torch.nn as nn

OFFSET_SCALE = 10.0  # targets are (box - last_box) / [W, H, W, H] * OFFSET_SCALE


class ConstantVelocity:
    """Trajectory baseline: average box velocity over the last `k` observed frames, extrapolated."""

    def __init__(self, k=5):
        self.k = k

    def predict(self, obs_boxes, pred_len):
        v = (obs_boxes[:, -1] - obs_boxes[:, -1 - self.k]) / self.k
        steps = np.arange(1, pred_len + 1)[None, :, None]
        return obs_boxes[:, -1][:, None, :] + v[:, None, :] * steps


class CueNet(nn.Module):
    def __init__(self, dims, groups, rnn="lstm", fusion="gated", emb=64, hidden=128, K=1,
                 pred_len=45, dec_cond=False, dropout=0.2):
        super().__init__()
        self.groups, self.fusion, self.K, self.pred_len, self.dec_cond = groups, fusion, K, pred_len, dec_cond
        self.enc = nn.ModuleDict({g: nn.Sequential(nn.Linear(dims[g], emb), nn.ReLU(), nn.Dropout(dropout),
                                                   nn.Linear(emb, emb), nn.ReLU()) for g in groups})
        G = len(groups)
        if fusion == "gated" and G > 1:
            self.gate = nn.Linear(G * emb, G)
        else:
            self.proj = nn.Linear(G * emb, emb)
        R = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = R(emb, hidden, batch_first=True)
        self.intent = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.mode_emb = nn.Embedding(K, 16)
        cond = 1 + 10 if dec_cond else 0
        self.dec_init = nn.Linear(hidden + 16, 2 * hidden)
        self.dec = nn.LSTMCell(4 + 16 + cond, hidden)
        self.dec_out = nn.Linear(hidden, 4)
        self.mode_logits = nn.Linear(hidden, K)
        self.hidden = hidden

    def encode(self, x):
        e = [self.enc[g](x[g]) for g in self.groups]                # each (B, T, emb)
        if hasattr(self, "gate"):
            a = torch.softmax(self.gate(torch.cat(e, -1)), -1)       # (B, T, G)
            z = (torch.stack(e, -1) * a.unsqueeze(2)).sum(-1)
        else:
            a = None
            z = self.proj(torch.cat(e, -1))
        out, _ = self.rnn(z)
        return out[:, -1], a

    def forward(self, x, traj=True):
        h, a = self.encode(x)
        logit = self.intent(h).squeeze(-1)
        res = dict(logit=logit, gates=a, mode_logits=self.mode_logits(h))
        if traj:
            B, K = h.shape[0], self.K
            m = self.mode_emb.weight.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
            hk = h.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            hc = torch.tanh(self.dec_init(torch.cat([hk, m], -1)))
            hd, cd = hc[:, :self.hidden].contiguous(), hc[:, self.hidden:].contiguous()
            prev = torch.zeros(B * K, 4, device=h.device)
            extra = [m]
            if self.dec_cond:
                p = torch.sigmoid(logit).detach().unsqueeze(1)
                ego_last = x["ego"][:, -1] if "ego" in x else torch.zeros(B, 10, device=h.device)
                extra += [p.repeat_interleave(K, 0), ego_last.repeat_interleave(K, 0)]
            outs = []
            for _ in range(self.pred_len):
                hd, cd = self.dec(torch.cat([prev] + extra, -1), (hd, cd))
                prev = self.dec_out(hd)
                outs.append(prev)
            res["traj"] = torch.stack(outs, 1).view(B, K, self.pred_len, 4)
        return res


def build_model(spec, dims, dropout=0.2):
    return CueNet(dims, spec["groups"], rnn=spec.get("rnn", "lstm"), fusion=spec.get("fusion", "gated"),
                  K=spec.get("K", 1), dec_cond=spec.get("dec_cond", False), dropout=dropout)
