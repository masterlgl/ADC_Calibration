import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from collections import defaultdict


class SinkhornDistance(nn.Module):
    """Sinkhorn OT with cosine distance computed only from means."""

    def __init__(self, eps=0.1, max_iter=200, reduction="none"):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, source_weights=None, target_weights=None):
        x = _as_2d(x)
        y = _as_2d(y).to(device=x.device, dtype=x.dtype)
        cost = 1.0 - F.normalize(x, dim=1).matmul(F.normalize(y, dim=1).T)

        mu = _normalize_weights(source_weights, x.shape[0], x.device, x.dtype)
        nu = _normalize_weights(target_weights, y.shape[0], x.device, x.dtype)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        for _ in range(self.max_iter):
            last_u = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self._modified_cost(cost, u, v), dim=-1)) + u
            v = self.eps * (
                torch.log(nu + 1e-8) - torch.logsumexp(self._modified_cost(cost, u, v).T, dim=-1)
            ) + v
            if (u - last_u).abs().sum().item() < 1e-1:
                break

        plan = torch.exp(self._modified_cost(cost, u, v))
        prob_plan = plan / plan.sum(dim=0, keepdim=True).clamp_min(1e-12)
        ot_cost = torch.sum(plan * cost)
        if self.reduction == "mean":
            ot_cost = ot_cost.mean()
        elif self.reduction == "sum":
            ot_cost = ot_cost.sum()
        return plan.detach().cpu(), prob_plan.detach().cpu(), ot_cost.detach().cpu()

    def _modified_cost(self, cost, u, v):
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


def cal_gaussmixture(feature, max_components=3, random_state=0, reg_covar=1e-6):
    feature_tensor = torch.as_tensor(feature)
    if feature_tensor.ndim != 2 or feature_tensor.shape[0] == 0:
        raise ValueError("feature must be a non-empty 2D tensor")

    n_components = min(max_components, feature_tensor.shape[0])
    feature_np = feature_tensor.detach().cpu().numpy()
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, reg_covar=reg_covar)
    gmm.fit(feature_np)
    labels = gmm.predict(feature_np)

    dtype = feature_tensor.dtype if feature_tensor.is_floating_point() else torch.float32
    device = feature_tensor.device
    means = torch.tensor(gmm.means_, dtype=dtype, device=device)
    covs = torch.tensor(gmm.covariances_, dtype=dtype, device=device)
    weights = torch.tensor(gmm.weights_, dtype=dtype, device=device)
    counts = torch.tensor(np.bincount(labels, minlength=n_components), dtype=dtype, device=device)

    return {
        "K": n_components,
        "means": means,
        "covs": covs,
        "weight": weights,
        "count": counts,
        "weight_mean": torch.sum(means * weights.unsqueeze(1), dim=0),
    }


def cal_statistic(dataloader, model, data_num_N):
    model.eval()
    device = next(model.parameters()).device
    feature_list, label_list = [], []

    with torch.no_grad():
        for data, gt, *_ in dataloader:
            _, feature = model.forward_with_feature(data.to(device).float())
            feature_list.append(feature.detach().cpu())
            label_list.append(gt.view(-1).detach().cpu())

    feature = torch.cat(feature_list, dim=0)
    label = torch.cat(label_list, dim=0)
    feature_per_class = defaultdict()
    cls_statistic = defaultdict(dict)

    for cls in range(len(data_num_N)):
        cls_feature = feature[torch.where(label == cls)[0]]
        if cls_feature.shape[0] == 0:
            raise ValueError(f"class {cls} has no feature")
        gmm = cal_gaussmixture(cls_feature)
        feature_per_class[cls] = cls_feature
        cls_statistic[cls] = {"mean": gmm["weight_mean"], "gmm": gmm}

    return cls_statistic, feature_per_class



def OT_calibrate(
    cls_stastic,
    feature_per_class,
    head_index,
    tail_index,
    data_num_N,
    samples_per_tail=500,
    tail_mean_weight=0.8,
    cov_reg=0.1,
):
    head_classes = _to_int_list(head_index)
    tail_classes = _to_int_list(tail_index)
    first_feature = feature_per_class[head_classes[0]]
    device = first_feature.device
    dtype = first_feature.dtype

    head_means = torch.stack([_class_mean(cls_stastic, cls, device, dtype) for cls in head_classes], dim=0)
    tail_means = torch.stack([_class_mean(cls_stastic, cls, device, dtype) for cls in tail_classes], dim=0)
    head_weights = _class_count_weights(data_num_N, head_classes, device, dtype)
    tail_weights = _class_count_weights(data_num_N, tail_classes, device, dtype, inverse=True)

    ot = SinkhornDistance(eps=0.1, max_iter=200).to(device)
    _, global_prob, _ = ot(head_means, tail_means, head_weights, tail_weights)
    global_prob = global_prob.to(device=device, dtype=dtype)

    finetune_features, finetune_labels = [], []
    for cls in head_classes:
        cls_feature = feature_per_class[cls]
        finetune_features.append(cls_feature)
        finetune_labels.append(torch.full((cls_feature.shape[0], 1), cls, dtype=torch.long, device=cls_feature.device))

    origin_features, origin_labels = [], []
    for cls in sorted(_to_int_list(feature_per_class.keys())):
        cls_feature = feature_per_class[cls]
        origin_features.append(cls_feature)
        origin_labels.append(torch.full((cls_feature.shape[0], 1), cls, dtype=torch.long, device=cls_feature.device))

    for tail_pos, tail_cls in enumerate(tail_classes):
        tail_gmm = _gmm_tensors(_gmm_stat(cls_stastic, tail_cls), device, dtype)
        t_means, t_covs, t_weights, t_counts, tail_k = tail_gmm
        calibrated_means = torch.zeros_like(t_means)
        calibrated_covs = torch.zeros_like(t_covs)

        for head_pos, head_cls in enumerate(head_classes):
            head_gmm = _gmm_tensors(_gmm_stat(cls_stastic, head_cls), device, dtype)
            h_means, h_covs, _, h_counts, _ = head_gmm
            _, local_prob, _ = ot(h_means, t_means, h_counts, 1.0 / t_counts.clamp_min(1e-8))
            local_prob = local_prob.to(device=device, dtype=dtype)

            mean_by_tail_component = local_prob.T.matmul(h_means)
            cov_by_tail_component = torch.einsum("hk,hde->kde", local_prob, h_covs)
            weight = global_prob[head_pos, tail_pos]
            calibrated_means += weight * mean_by_tail_component
            calibrated_covs += weight * cov_by_tail_component

        new_means = tail_mean_weight * t_means + (1.0 - tail_mean_weight) * calibrated_means
        new_covs = _regularize_cov(calibrated_covs, cov_reg)
        sample_counts = _allocate_samples(t_weights.detach().cpu(), samples_per_tail)

        sampled = []
        for component_id, sample_count in enumerate(sample_counts[:tail_k]):
            if sample_count <= 0:
                continue
            dist = torch.distributions.MultivariateNormal(new_means[component_id], new_covs[component_id])
            sampled.append(dist.sample((sample_count,)))
        new_tail_feature = torch.cat(sampled, dim=0)

        finetune_features.append(new_tail_feature)
        finetune_labels.append(torch.full((new_tail_feature.shape[0], 1), tail_cls, dtype=torch.long, device=device))

    return (
        torch.cat(finetune_features, dim=0),
        torch.cat(finetune_labels, dim=0).cpu(),
        torch.cat(origin_features, dim=0),
        torch.cat(origin_labels, dim=0).cpu(),
    )


def _as_2d(x):
    x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError("SinkhornDistance expects 1D or 2D tensors")
    return x


def _normalize_weights(weights, size, device, dtype):
    if weights is None:
        weights = torch.ones(size, device=device, dtype=dtype)
    else:
        weights = torch.as_tensor(weights, device=device, dtype=dtype)
    weights = weights.reshape(-1).clamp_min(1e-8)
    if weights.numel() != size:
        raise ValueError(f"expected {size} weights, got {weights.numel()}")
    return weights / weights.sum()


def _to_int_list(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    return [int(value) for value in values]


def _gmm_stat(cls_statistic, cls):
    stat = cls_statistic[int(cls)]
    if isinstance(stat, dict):
        return stat.get("gmm", stat)
    return stat[5]


def _gmm_tensors(gmm, device=None, dtype=None):
    means = torch.as_tensor(gmm["means"], device=device, dtype=dtype)
    covs = torch.as_tensor(gmm["covs"], device=device, dtype=dtype)
    weights = torch.as_tensor(gmm["weight"], device=device, dtype=dtype).reshape(-1)
    counts = torch.as_tensor(gmm.get("count", weights), device=device, dtype=dtype).reshape(-1)
    k = int(gmm.get("K", means.shape[0]))

    if counts.numel() != k:
        counts = weights * max(float(counts.sum().item()), 1.0)
    return means, covs, weights.clamp_min(1e-8), counts.clamp_min(1e-8), k


def _class_mean(cls_statistic, cls, device=None, dtype=None):
    means, _, weights, _, _ = _gmm_tensors(_gmm_stat(cls_statistic, cls), device, dtype)
    weights = weights / weights.sum()
    return torch.sum(means * weights.unsqueeze(1), dim=0)


def _class_count_weights(data_num_N, classes, device, dtype, inverse=False):
    counts = torch.as_tensor(data_num_N, device=device, dtype=dtype)[classes]
    counts = counts.clamp_min(1e-8)
    return 1.0 / counts if inverse else counts


def _regularize_cov(covs, cov_reg):
    dim = covs.shape[-1]
    eye = torch.eye(dim, dtype=covs.dtype, device=covs.device)
    covs = 0.5 * (covs + covs.transpose(-1, -2))
    return covs + cov_reg * eye.unsqueeze(0)


def _allocate_samples(weights, total):
    weights = _normalize_weights(weights, len(weights), torch.device("cpu"), torch.float64)
    expected = weights * int(total)
    counts = torch.floor(expected).long()
    remainder = int(total) - int(counts.sum().item())
    if remainder > 0:
        order = torch.argsort(expected - counts, descending=True)
        counts[order[:remainder]] += 1
    return counts.tolist()
