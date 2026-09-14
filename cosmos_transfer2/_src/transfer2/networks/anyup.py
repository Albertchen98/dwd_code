from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional, Tuple
from functools import lru_cache


def window2d(
        low_res: int | Tuple[int, int],
        high_res: int | Tuple[int, int],
        ratio: float,
        *,
        device: str = "cpu"
) -> torch.Tensor:
    # unpack
    if isinstance(high_res, int):
        H = W = high_res
    else:
        H, W = high_res
    if isinstance(low_res, int):
        Lh = Lw = low_res
    else:
        Lh, Lw = low_res

    # pixel-centers in [0,1)
    r_pos = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H  # (H,)
    c_pos = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W  # (W,)
    pos_r, pos_c = torch.meshgrid(r_pos, c_pos, indexing="ij")  # (H,W)

    # clamp before scaling
    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    # quantise symmetrically
    r0 = (r_lo * Lh).floor().long()  # inclusive start
    r1 = (r_hi * Lh).ceil().long()  # exclusive end
    c0 = (c_lo * Lw).floor().long()
    c1 = (c_hi * Lw).ceil().long()

    return torch.stack([r0, r1, c0, c1], dim=2)


@lru_cache
def compute_attention_mask(high_res_h, high_res_w, low_res_h, low_res_w, window_size_ratio, device="cpu"):
    h, w = high_res_h, high_res_w
    h_, w_ = low_res_h, low_res_w

    windows = window2d(
        low_res=(h_, w_),
        high_res=(h, w),
        ratio=window_size_ratio,
        device=device
    )

    q = h * w  # number of high-res query locations

    # flatten window bounds: (q, 1)
    r0 = windows[..., 0].reshape(q, 1)
    r1 = windows[..., 1].reshape(q, 1)  # exclusive
    c0 = windows[..., 2].reshape(q, 1)
    c1 = windows[..., 3].reshape(q, 1)  # exclusive

    # row / column indices on low-res grid
    rows = torch.arange(h_, device=device)  # (h_,)
    cols = torch.arange(w_, device=device)  # (w_,)

    row_ok = (rows >= r0) & (rows < r1)  # (q, h_)
    col_ok = (cols >= c0) & (cols < c1)  # (q, w_)

    # broadcast to (q, h_, w_) and flatten last two dims
    attention_mask = (row_ok.unsqueeze(2) & col_ok.unsqueeze(1)) \
        .reshape(q, h_ * w_).to(dtype=torch.bool)

    return ~attention_mask


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8,
                 pad_mode="zeros", norm_fn=None, activation_fn=nn.SiLU, use_conv_shortcut=False):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
            N(out_channels),
            activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

compute_basis_size = {"gauss_deriv": lambda order, mirror: ((order + 1) * (order + 2)) // (1 if mirror else 2)}

def herme_vander_torch(z, m):
    He0 = z.new_ones(z.shape)
    if m == 0: return He0[:, None]
    H = [He0, z]
    for n in range(1, m):
        H.append(z * H[-1] - n * H[-2])
    return torch.stack(H, 1)


def gauss_deriv(max_order, device, dtype, kernel_size, sigma=None, include_negations=False, scale_magnitude=True):
    sigma = (kernel_size // 2) / 1.645 if sigma is None else sigma
    if kernel_size % 2 == 0: raise ValueError("ksize must be odd")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    z = x / sigma
    g = torch.exp(-0.5 * z ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    He = herme_vander_torch(z, max_order)
    derivs_1d = [(((-1) ** n) / (sigma ** n) if scale_magnitude else (-1) ** n) * He[:, n] * g for n in
                 range(max_order + 1)]
    bank = []
    for o in range(max_order + 1):
        for i in range(o + 1):
            K = torch.outer(derivs_1d[o - i], derivs_1d[i])
            bank.append(K)
            if include_negations: bank.append(-K)
    return torch.stack(bank, 0)


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if init_gaussian_derivatives:
            # find smallest order that gives at least out_channels basis functions
            order = 0
            while compute_basis_size["gauss_deriv"](order, False) < out_channels:
                order += 1
            print(f"FeatureUnification: initializing with Gaussian derivative basis of order {order}")
            self.basis = nn.Parameter(
                gauss_deriv(
                    order, device='cpu', dtype=torch.float32, kernel_size=kernel_size, scale_magnitude=False
                )[:out_channels, None]
            )
        else:
            self.basis = nn.Parameter(
                torch.randn(out_channels, 1, kernel_size, kernel_size)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, h, w)
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(F.pad(mask, (p, p, p, p), value=0), torch.ones(1, 1, k, k, device=x.device))
        return x / denom  # (B, out_channels*C, H, W)

class CrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads,
                 q_chunk_size: Optional[int] = None,
                 store_attn: bool = False):
        super().__init__()
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_chunk_size = q_chunk_size
        self.store_attn = store_attn
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        # 2D: (tgt_len, src_len), 3D: (B*num_heads or B, tgt_len, src_len)
        if mask.dim() == 2:
            return mask[start:end, :]
        elif mask.dim() == 3:
            return mask[:, start:end, :]
        else:
            raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None,
                q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None):
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn

        val = key

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Fast path: no chunking
        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val,
                                     average_attn_weights=True,
                                     attn_mask=mask)
            features = torch.einsum("b i j, b j d -> b i d", attn, value)
            return features, (attn if store_attn else None)

        # Chunked over the query length (tgt_len)
        B, Q, _ = query.shape
        outputs = []
        attns = [] if store_attn else None

        for start in range(0, Q, q_chunk_size):
            end = min(start + q_chunk_size, Q)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)

            # We ignore the MHA output as in JAFAR:
            # use the averaged attention to weight the unprojected V.
            _, attn_chunk = self.attention(q_chunk, key, val,
                                           average_attn_weights=True,
                                           attn_mask=mask_chunk)
            out_chunk = torch.einsum("b i j, b j d -> b i d", attn_chunk, value)
            outputs.append(out_chunk)
            if store_attn:
                attns.append(attn_chunk)

        features = torch.cat(outputs, dim=1)
        attn_scores = torch.cat(attns, dim=1) if store_attn else None
        return features, attn_scores


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1,
                 q_chunk_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(
            qk_dim, num_heads,
            q_chunk_size=q_chunk_size
        )
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None, store_attn: Optional[bool] = None, vis_attn=False,
                **kwargs):
        store_attn = store_attn or vis_attn
        q = self.conv2d(q)
        if self.window_ratio > 0:
            attn_mask = compute_attention_mask(
                *q.shape[-2:], *k.shape[-2:], window_size_ratio=self.window_ratio
            ).to(q.device)
        else:
            attn_mask = None
        b, _, h, w = q.shape
        _, _, h_k, w_k = k.shape
        c = v.shape[1]
        q = q.permute(0, 2, 3, 1).view(b, h * w, -1)
        k = k.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)
        v = v.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)

        features, attn = self.cross_attn(q, k, v, mask=attn_mask,
                                         q_chunk_size=q_chunk_size,
                                         store_attn=store_attn)
        features = features.view(b, h, w, c).permute(0, 3, 1, 2)

        return features

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()

def create_coordinate(h, w, start=0.0, end=1.0, device=None, dtype=None):
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy), -1).view(1, h * w, 2)


class AnyUp(nn.Module):
    def __init__(
            self,
            input_dim=3,
            qk_dim=128,
            kernel_size=1,
            kernel_size_lfu=5,
            window_ratio=0.1,
            num_heads=4,
            init_gaussian_derivatives=False,
            **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm,
                             activation_fn=nn.SiLU)

        # Encoders
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(None, 1, first_layer_k=kernel_size_lfu,
                                                       init_gaussian_derivatives=init_gaussian_derivatives)

        # Cross-attention
        self.cross_decode = CrossAttentionBlock(qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio)
        self.aggregation = self._make_encoder(2 * qk_dim, 3)

        # RoPE for (H*W, C)
        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2d(in_ch, self.qk_dim, k, padding=k // 2, padding_mode="reflect", bias=False)
            if first_layer_k == 0 else
            LearnedFeatureUnification(self.qk_dim, first_layer_k, init_gaussian_derivatives=init_gaussian_derivatives)
        )
        blocks = [ResBlock(self.qk_dim, self.qk_dim, **self._rb_args) for _ in range(layers)]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, out_size, vis_attn=False, q_chunk_size=None):
        b, c, h, w = feats.shape

        # Q
        q = F.adaptive_avg_pool2d(self.query_encoder(enc_img), output_size=out_size)

        # K
        k = F.adaptive_avg_pool2d(self.key_encoder(enc_img), output_size=(h, w))
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)

        # V
        v = feats

        return self.cross_decode(q, k, v, vis_attn=vis_attn, q_chunk_size=q_chunk_size)

    def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
        output_size = output_size if output_size is not None else image.shape[-2:]
        enc = self.image_encoder(image)
        h = enc.shape[-2]
        coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)
        return self.upsample(enc, features, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
