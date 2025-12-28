import math
from typing import Sequence

import pandas as pd
from utils import iter_progress

try:
    import torch
except ImportError:
    torch = None

# shrinking the range to only the cluster counts I "realistically" need instead of something like [1,2,3, ...]
DEFAULT_COMPONENT_GRID = [4, 8, 12, 16, 20]

def ensure_torch_tensor(data, device: str, dtype=torch.float32):
    if torch is None:
        raise ImportError("PyTorch is required for GPU-based operations.")
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, pd.DataFrame):
        array = data.values
    else:
        array = data
    return torch.as_tensor(array, device=device, dtype=dtype)


class TorchPowerTransformer:
    """Torch implementation of sklearn's PowerTransformer (Box-Cox)."""

    def __init__(self, standardize: bool = True):
        if torch is None:
            raise ImportError("PyTorch is required for TorchPowerTransformer.")
        self.standardize = standardize
        self.device = None
        self.dtype = None
        self.lambdas_ = None
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X: torch.Tensor):
        if (X <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values. Apply a positive shift before fitting.")
        self.device = X.device
        self.dtype = X.dtype
        _, n_features = X.shape
        lambdas = torch.empty(n_features, device=self.device, dtype=self.dtype)
        means = torch.empty_like(lambdas)
        scales = torch.empty_like(lambdas)
        for idx in range(n_features):
            column = X[:, idx]
            lam = self._find_lambda(column)
            lambdas[idx] = lam
            transformed = self._boxcox(column, lam)
            col_mean = transformed.mean()
            if self.standardize:
                col_std = torch.std(transformed, unbiased=False)
                col_std = torch.clamp(col_std, min=1e-12)
            else:
                col_std = torch.ones((), device=self.device, dtype=self.dtype)
            means[idx] = col_mean
            scales[idx] = col_std

        self.lambdas_ = lambdas
        self.mean_ = means
        self.scale_ = scales
        self.n_features_in_ = n_features
        return self

    def fit_transform(self, X: torch.Tensor):
        return self.fit(X).transform(X)

    def transform(self, X: torch.Tensor):
        self._check_is_fitted()
        transformed_cols = []
        for idx in range(self.n_features_in_):
            column = X[:, idx]
            lam = self.lambdas_[idx]
            col_trans = self._boxcox(column, lam)
            if self.standardize:
                col_trans = (col_trans - self.mean_[idx]) / self.scale_[idx]
            transformed_cols.append(col_trans)
        return torch.stack(transformed_cols, dim=1)

    def inverse_transform(self, X: torch.Tensor):
        self._check_is_fitted()
        inverted_cols = []
        for idx in range(self.n_features_in_):
            column = X[:, idx]
            if self.standardize:
                column = column * self.scale_[idx] + self.mean_[idx]
            lam = self.lambdas_[idx]
            inverted_cols.append(self._boxcox_inverse(column, lam))
        return torch.stack(inverted_cols, dim=1)

    def _boxcox(self, data: torch.Tensor, lam: torch.Tensor):
        eps = 1e-8
        if torch.abs(lam) < eps:
            return torch.log(data)
        return (torch.pow(data, lam) - 1.0) / lam

    def _boxcox_inverse(self, data: torch.Tensor, lam: torch.Tensor):
        eps = 1e-8
        if torch.abs(lam) < eps:
            return torch.exp(data)
        return torch.pow(lam * data + 1.0, 1.0 / lam)

    def _boxcox_llf(self, lam_value: float, data: torch.Tensor) -> float:
        lam = torch.tensor(lam_value, device=data.device, dtype=data.dtype)
        transformed = self._boxcox(data, lam)
        centered = transformed - transformed.mean()
        var = torch.mean(centered * centered)
        var = torch.clamp(var, min=1e-12)
        llf = (lam - 1.0) * torch.sum(torch.log(data)) - 0.5 * data.shape[0] * torch.log(var)
        return float(llf.item())

    def _find_lambda(self, data: torch.Tensor):
        lower, upper = -2.0, 2.0
        tol = 1e-5
        invphi = (math.sqrt(5) - 1) / 2
        invphi2 = (3 - math.sqrt(5)) / 2
        a, b = lower, upper
        h = b - a
        if h <= tol:
            return torch.tensor((a + b) / 2.0, device=data.device, dtype=data.dtype)
        c = a + invphi2 * h
        d = a + invphi * h
        yc = self._boxcox_llf(c, data)
        yd = self._boxcox_llf(d, data)
        while abs(b - a) > tol:
            if yc > yd:
                b, d, yd = d, c, yc
                h = invphi * (b - a)
                c = a + invphi2 * (b - a)
                yc = self._boxcox_llf(c, data)
            else:
                a, c, yc = c, d, yd
                h = invphi * (b - a)
                d = a + invphi * (b - a)
                yd = self._boxcox_llf(d, data)
        lam_est = (a + b) / 2.0
        return torch.tensor(lam_est, device=data.device, dtype=data.dtype)

    def _check_is_fitted(self):
        if self.lambdas_ is None:
            raise RuntimeError("TorchPowerTransformer must be fitted before calling transform.")


def prepare_boxcox_features_gpu(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    boxcox_shift: float,
    device: str = "cuda",
    dtype=None,
):
    if torch is None:
        raise ImportError("PyTorch is required for GPU-based Box-Cox preprocessing.")
    X = df[feature_cols].copy()
    X_boxcox_input = X + boxcox_shift
    constant_cols = X_boxcox_input.columns[X_boxcox_input.nunique() <= 1].tolist()
    constant_values = {col: X[col].iloc[0] for col in constant_cols}
    train_cols = [col for col in feature_cols if col not in constant_cols]
    if not train_cols:
        raise ValueError("No features left after removing constant columns for GPU pipeline.")
    torch_device = torch.device(device)
    tensor_dtype = dtype or torch.float64
    tensor_input = torch.as_tensor(
        X_boxcox_input[train_cols].values,
        device=torch_device,
        dtype=tensor_dtype,
    )
    transformer = TorchPowerTransformer()
    transformed = transformer.fit_transform(tensor_input)
    return transformed, transformer, constant_cols, constant_values, train_cols


def invert_latent_samples_gpu(
    latent_samples,
    transformer: TorchPowerTransformer,
    feature_names: Sequence[str],
    boxcox_shift: float,
    device: str = "cuda",
    index=None,
):
    if torch is None:
        raise ImportError("PyTorch is required for GPU-based inverse Box-Cox transform.")
    latent_tensor = torch.as_tensor(latent_samples, device=torch.device(device), dtype=transformer.dtype)
    positive = transformer.inverse_transform(latent_tensor)
    positive = positive - boxcox_shift
    positive = torch.nan_to_num(positive, nan=0.0, posinf=0.0, neginf=0.0)
    positive = torch.clamp(positive, min=0.0)
    synthetic_df = pd.DataFrame(positive.cpu().numpy(), columns=feature_names)
    if index is not None:
        synthetic_df.index = index
    return synthetic_df

class TorchGaussianMixture:
    def __init__(self, weights, means, covariances, log_likelihood, device, dtype):
        self.weights_ = weights.detach().clone()
        self.means_ = means.detach().clone()
        self.covariances_ = covariances.detach().clone()
        self.log_likelihood_ = log_likelihood
        self.n_components = self.weights_.shape[0]
        self.n_features_in_ = self.means_.shape[1]
        self._device = device
        self._dtype = dtype

    def _require_torch(self):
        if torch is None:
            raise ImportError("PyTorch is required to sample from a TorchGaussianMixture instance.")

    def sample(self, n_samples: int, random_state: int | None = None):
        self._require_torch()
        if random_state is None:
            torch.seed()
        else:
            torch.manual_seed(int(random_state))

        device = self._device
        dtype = self._dtype
        weights = self.weights_.to(device=device, dtype=dtype)
        means = self.means_.to(device=device, dtype=dtype)
        covariances = self.covariances_.to(device=device, dtype=dtype)

        component_ids = torch.multinomial(weights, n_samples, replacement=True)
        samples = torch.empty((n_samples, self.n_features_in_), device=device, dtype=dtype)
        for comp in range(self.n_components):
            mask = component_ids == comp
            count = int(mask.sum())
            if count == 0:
                continue
            dist = torch.distributions.MultivariateNormal(means[comp], covariance_matrix=covariances[comp])
            samples[mask] = dist.sample((count,))

        return samples.cpu().numpy(), component_ids.cpu().numpy()


def fit_best_gmm_gpu(
    X_boxcox,
    component_grid=DEFAULT_COMPONENT_GRID,
    random_state: int = 42,
    max_iter: int = 400,
    tol: float = 1e-3,
    n_init: int = 2,
    reg_covar: float = 5e-3,
    device: str = "cuda",
    batch_size: int | None = None,
    dtype=None,
    use_kmeans_init: bool = True,
    kmeans_iters: int = 10,
):
    if torch is None:
        raise ImportError("PyTorch is required for GPU-based GMM training. Please install torch with CUDA support.")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but not available. Check your PyTorch installation or GPU drivers.")

    torch_device = torch.device(device)
    tensor_dtype = dtype or torch.float64
    data = ensure_torch_tensor(X_boxcox, torch_device, dtype=tensor_dtype)
    n_samples, n_features = data.shape
    if batch_size is None or batch_size <= 0:
        em_batch_size = n_samples
    else:
        em_batch_size = batch_size
    kmeans_batch_size = min(em_batch_size, n_samples)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(random_state)

    data_dtype = data.dtype
    eye = torch.eye(n_features, device=torch_device, dtype=data_dtype)
    log_2pi = torch.log(torch.tensor(2.0 * math.pi, device=torch_device, dtype=data_dtype))

    def assign_clusters(centers):
        labels = torch.empty(n_samples, dtype=torch.long, device=torch_device)
        for start in range(0, n_samples, kmeans_batch_size):
            end = min(start + kmeans_batch_size, n_samples)
            batch = data[start:end]
            distances = torch.cdist(batch, centers)
            labels[start:end] = torch.argmin(distances, dim=1)
        return labels

    def compute_covariances_from_labels(labels, means):
        covariances = torch.empty((means.shape[0], n_features, n_features), device=torch_device, dtype=data_dtype)
        counts = torch.bincount(labels, minlength=means.shape[0]).to(device=torch_device, dtype=data_dtype)
        total = torch.clamp(counts.sum(), min=torch.finfo(data_dtype).eps)
        weights = counts / total
        for comp in range(means.shape[0]):
            mask = labels == comp
            if mask.any():
                centered = data[mask] - means[comp]
                cov = centered.t().mm(centered) / torch.clamp(counts[comp], min=1.0)
                covariances[comp] = cov + reg_covar * eye
            else:
                covariances[comp] = eye.clone()
                weights[comp] = torch.tensor(1.0 / means.shape[0], device=torch_device, dtype=data_dtype)
        weights = weights / torch.clamp(weights.sum(), min=torch.finfo(data_dtype).eps)
        return weights, covariances

    def kmeans_initialize_params(k: int):
        indices = torch.randint(0, n_samples, (k,), device=torch_device, generator=generator)
        centers = data[indices].clone()
        for _ in range(max(1, kmeans_iters)):
            labels = assign_clusters(centers)
            updated = centers.clone()
            for comp in range(k):
                mask = labels == comp
                if mask.any():
                    updated[comp] = data[mask].mean(dim=0)
                else:
                    rand_idx = torch.randint(0, n_samples, (1,), device=torch_device, generator=generator)
                    updated[comp] = data[rand_idx]
            centers = updated
        labels = assign_clusters(centers)
        weights, covariances = compute_covariances_from_labels(labels, centers)
        return weights, centers.clone(), covariances

    def initialize_params(k: int):
        if use_kmeans_init:
            try:
                return kmeans_initialize_params(k)
            except RuntimeError:
                pass
        indices = torch.randint(0, n_samples, (k,), device=torch_device, generator=generator)
        means = data[indices].clone()
        covariances = eye.unsqueeze(0).repeat(k, 1, 1)
        weights = torch.full((k,), 1.0 / k, device=torch_device, dtype=data_dtype)
        return weights, means, covariances

    def estimate_log_gaussian_prob_batch(batch, means, covariances):
        log_probs = []
        for comp in range(means.shape[0]):
            L = torch.linalg.cholesky(covariances[comp])
            diff = batch - means[comp]
            sol = torch.cholesky_solve(diff.unsqueeze(-1), L)
            maha = torch.sum(diff.unsqueeze(-1) * sol, dim=1).squeeze(-1)
            log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
            norm = batch.shape[1] * log_2pi + log_det
            log_probs.append(-0.5 * (norm + maha))
        return torch.stack(log_probs, dim=1)

    def run_em(k: int):
        best_ll = -float("inf")
        best = None
        init_iter = iter_progress(range(max(1, n_init)), desc=f"GPU init (k={k})", leave=False)
        for init in init_iter:
            weights, means, covariances = initialize_params(k)
            prev_ll = None
            em_iter = iter_progress(range(max_iter), desc=f"GPU EM (k={k}, init={int(init) + 1})", leave=False)
            for _ in em_iter:
                log_weights = torch.log(torch.clamp(weights, min=torch.finfo(weights.dtype).eps))
                nk = torch.zeros(k, device=torch_device, dtype=data_dtype)
                first_moment = torch.zeros((k, n_features), device=torch_device, dtype=data_dtype)
                second_moment = torch.zeros((k, n_features, n_features), device=torch_device, dtype=data_dtype)
                total_ll = torch.tensor(0.0, device=torch_device, dtype=data_dtype)

                for start in range(0, n_samples, em_batch_size):
                    end = min(start + em_batch_size, n_samples)
                    batch = data[start:end]
                    batch_log_prob = estimate_log_gaussian_prob_batch(batch, means, covariances)
                    log_prob = batch_log_prob + log_weights
                    log_prob_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
                    resp = torch.exp(log_prob - log_prob_norm)
                    total_ll += log_prob_norm.sum()
                    nk += resp.sum(dim=0)
                    first_moment += resp.t() @ batch
                    second_moment += torch.einsum("bk,bf,bg->kfg", resp, batch, batch)

                nk = torch.clamp(nk, min=torch.finfo(data_dtype).eps)
                weights = nk / nk.sum()
                means = first_moment / nk.unsqueeze(1)
                mean_outer = means.unsqueeze(-1) * means.unsqueeze(-2)
                covariances = second_moment / nk.view(-1, 1, 1) - mean_outer
                covariances = covariances + reg_covar * eye
                covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
                ll_value = float(total_ll.item())

                if prev_ll is not None and abs(ll_value - prev_ll) <= tol * n_samples:
                    break
                prev_ll = ll_value

            if ll_value > best_ll:
                best_ll = ll_value
                best = (weights.clone(), means.clone(), covariances.clone())
        if best is None:
            raise RuntimeError("GPU GMM training failed to converge for k=%d" % k)
        return best, best_ll

    bic_records = []
    best_model = None
    best_bic = float("inf")
    component_iter = iter_progress(component_grid, desc="GPU GMM components")
    for k in component_iter:
        (weights, means, covariances), ll_value = run_em(k)
        num_params = k - 1 + k * n_features + k * n_features * (n_features + 1) / 2
        bic = -2 * ll_value + num_params * math.log(n_samples)
        bic_records.append({"n_components": k, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_model = TorchGaussianMixture(weights, means, covariances, ll_value, torch_device, data_dtype)

    if best_model is None:
        raise RuntimeError("GPU GMM training did not yield a valid model.")
    bic_df = pd.DataFrame(bic_records)
    return best_model, bic_df
