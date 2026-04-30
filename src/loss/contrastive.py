import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).

    Robust to class imbalance: every same-label pair in the batch contributes,
    so minority-class anchors still get gradient signal from any positives present.
    Samples with label == -1 are ignored (treated as having no positives).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D], labels: [B]
        valid = labels >= 0
        if valid.sum() < 2:
            return features.sum() * 0.0

        f = F.normalize(features[valid], dim=1)
        y = labels[valid].view(-1, 1)
        N = f.size(0)

        sim = torch.matmul(f, f.T) / self.temperature
        # for numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(N, dtype=torch.bool, device=f.device)
        pos_mask = (y == y.T) & ~self_mask

        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        anchors = pos_count > 0
        if anchors.sum() == 0:
            return features.sum() * 0.0

        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[anchors] / pos_count[anchors].float()
        return -mean_log_prob_pos.mean()


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss (Hermans et al. 2017).

    Better than SupCon for small batches: for each anchor we mine the
    hardest positive and hardest negative present in the batch and pull/push
    only those. Works as long as a batch contains >=1 positive and >=1
    negative for the anchor's class — typical for binary deepfake batches.
    Samples with label == -1 are skipped.
    """

    def __init__(self, margin: float = 0.3, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels >= 0
        if valid.sum() < 2:
            return features.sum() * 0.0

        f = features[valid]
        if self.normalize:
            f = F.normalize(f, dim=1)
        y = labels[valid].view(-1, 1)
        N = f.size(0)

        # pairwise squared euclidean distances
        dist = torch.cdist(f, f, p=2)

        same = (y == y.T)
        diff = ~same
        eye = torch.eye(N, dtype=torch.bool, device=f.device)
        pos_mask = same & ~eye

        has_pos = pos_mask.any(dim=1)
        has_neg = diff.any(dim=1)
        anchors = has_pos & has_neg
        if anchors.sum() == 0:
            return features.sum() * 0.0

        # hardest positive: max dist among positives
        pos_d = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
        # hardest negative: min dist among negatives
        neg_d = dist.masked_fill(~diff, float("inf")).min(dim=1).values

        loss = F.relu(pos_d[anchors] - neg_d[anchors] + self.margin)
        return loss.mean()


class InfoNCEConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nce_logits, target_weights):
        """
        nce_logits: [B, SeqLen] (Cosine Similarity / Temperature)
        target_weights: [B, SeqLen] (Attention Weights from Pooler)
        """
        log_probs = F.log_softmax(nce_logits, dim=1)
        loss = -torch.sum(target_weights.detach() * log_probs, dim=1).mean()
        return loss