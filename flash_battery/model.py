from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .distributed_mode import is_main_process


def build_model(args):
    model_params = args.dit_model_params
    model_type = model_params.get("model", "Attention")
    model_mapping = {
        "DiT1D": DiT1D,
        "Attention": AttentionDiT,
    }
    if model_type not in model_mapping:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Available: {list(model_mapping.keys())}"
        )

    if model_type == "DiT1D":
        return DiT1D(
            seq_len=model_params.get("seq_len", 2048),
            patch_size=model_params.get("patch_size", 64),
            in_channels=model_params.get("in_channels", 3),
            hidden_dim=model_params.get("hidden_size", 384),
            depth=model_params.get("depth", 12),
            num_heads=model_params.get("num_heads", 6),
            mlp_ratio=model_params.get("mlp_ratio", 4.0),
            protocol_dim=model_params.get("protocol_dim", 100),
        )

    return AttentionDiT(
        seq_len=model_params.get("seq_len", 2048),
        patch_size=model_params.get("patch_size", 32),
        in_channels=model_params.get("in_channels", 3),
        out_channels=model_params.get("out_channels"),
        hidden_dim=model_params.get("hidden_size", 384),
        depth=model_params.get("depth", 12),
        num_heads=model_params.get("num_heads", 6),
        mlp_ratio=model_params.get("mlp_ratio", 4.0),
        protocol_dim=model_params.get("protocol_dim", 100),
    )


def load_model(args, model_without_ddp) -> None:
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model_without_ddp.load_state_dict(state_dict, strict=False)
    if is_main_process():
        print(f"Loaded checkpoint from {args.resume}")


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_length, _ = x.shape
        return x + self.pe[:, :seq_length]


class TimeStepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ScalarConditionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def embedding(x: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=x.device)
            / half
        )
        args = x[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(x, self.frequency_embedding_size))


class ProtocolEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, protocols: torch.Tensor) -> torch.Tensor:
        return self.mlp(protocols)


class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"Sequence length {seq_len} must be divisible by patch size {patch_size}"
        )
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        embedding_out = self.modulation(emb).unsqueeze(1)
        return (self.norm(x),) + embedding_out.chunk(6, dim=-1)


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ada_ln = AdaLNZero(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        (
            x_normed,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_ln(x, emb)

        x_modulated_msa = x_normed * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_modulated_msa, x_modulated_msa, x_modulated_msa)
        x = x + gate_msa * attn_out

        x_modulated_mlp = self.ada_ln.norm(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_modulated_mlp)
        x = x + gate_mlp * mlp_out
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


class DiT1D(nn.Module):
    def __init__(
        self,
        seq_len: int = 2048,
        patch_size: int = 64,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        protocol_dim: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed1D(seq_len, patch_size, in_channels, hidden_dim)
        self.pos_embed = SinusoidalPositionalEmbedding(
            hidden_dim, max_seq_length=self.patch_embed.num_patches
        )
        self.t_embedder = TimeStepEmbedder(hidden_dim)
        self.soh_embedder = ScalarConditionEmbedder(hidden_dim)
        self.y_embedder = ProtocolEmbedder(protocol_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock1D(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        torch.nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.ada_ln.modulation.weight, 0)
            nn.init.constant_(block.ada_ln.modulation.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation[1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation[1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, _ = x.shape
        x = x.view(batch, patches, self.in_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        return x.flatten(2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        soh: torch.Tensor,
        protocols: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        del extra
        t_emb = self.t_embedder(t.expand(x.shape[0]))
        soh_emb = self.soh_embedder(soh)
        y_emb = self.y_embedder(protocols)
        cond_emb = t_emb + soh_emb + y_emb

        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, cond_emb)
        x = self.final_layer(x, cond_emb)
        return self.unpatchify(x)


class IntraPatchSelfAttention(nn.Module):
    def __init__(self, patch_size: int, d_model: int, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_length=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        num_patches = seq_len // self.patch_size
        x_patch = x.reshape(batch * num_patches, self.patch_size, dim)
        x_patch = self.pos_emb(x_patch)
        attn_out, _ = self.attn(x_patch, x_patch, x_patch)
        return attn_out.reshape(batch, seq_len, dim)


class InterPatchAxialAttention(nn.Module):
    def __init__(self, patch_size: int, d_model: int, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_length=4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        num_patches = seq_len // self.patch_size
        x_reshaped = x.reshape(batch, num_patches, self.patch_size, dim)
        x_axial = x_reshaped.permute(0, 2, 1, 3).reshape(
            batch * self.patch_size, num_patches, dim
        )
        x_axial = self.pos_emb(x_axial)
        attn_out, _ = self.attn(x_axial, x_axial, x_axial)
        attn_out = attn_out.reshape(batch, self.patch_size, num_patches, dim)
        attn_out = attn_out.permute(0, 2, 1, 3)
        return attn_out.reshape(batch, seq_len, dim)


class FactorizedDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.ada_ln = AdaLNZero(hidden_dim)
        self.intra_attn = IntraPatchSelfAttention(patch_size, hidden_dim, num_heads)
        self.inter_attn = InterPatchAxialAttention(patch_size, hidden_dim, num_heads)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        (
            x_normed,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_ln(x, emb)
        x_modulated = x_normed * (1 + scale_msa) + shift_msa
        attn_out = self.intra_attn(x_modulated) + self.inter_attn(x_modulated)
        x = x + gate_msa * attn_out
        x_modulated_mlp = self.ada_ln.norm(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_modulated_mlp)
        x = x + gate_mlp * mlp_out
        return x


class SequenceFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None):
        if force_drop_ids is None:
            return self.embedding(labels)
        drop_labels = torch.where(
            force_drop_ids,
            torch.full_like(labels, self.num_classes),
            labels,
        )
        return self.embedding(drop_labels)


class AttentionDiT(nn.Module):
    def __init__(
        self,
        seq_len: int = 2048,
        patch_size: int = 32,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        protocol_dim: int = 100,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        out_channels = out_channels or in_channels
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_seq_length=seq_len)
        self.t_embedder = TimeStepEmbedder(hidden_dim)
        self.soh_embedder = ScalarConditionEmbedder(hidden_dim)
        self.y_embedder = ProtocolEmbedder(protocol_dim, hidden_dim)
        self.label_embedder = LabelEmbedder(num_classes=2, hidden_dim=hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FactorizedDiTBlock(hidden_dim, num_heads, patch_size, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = SequenceFinalLayer(hidden_dim, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embedder.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.ada_ln.modulation.weight, 0)
            nn.init.constant_(block.ada_ln.modulation.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation[1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation[1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        soh: torch.Tensor,
        protocols: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        t_emb = self.t_embedder(t.expand(x.shape[0]))
        soh_emb = self.soh_embedder(soh)
        y_emb = self.y_embedder(protocols)

        if extra is not None and "label" in extra:
            labels = extra["label"]
            drop_mask = extra.get("drop_mask")
            label_emb = self.label_embedder(labels, drop_mask)
        else:
            drop_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
            zeros = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            label_emb = self.label_embedder(zeros, drop_mask)

        cond_emb = t_emb + soh_emb + y_emb + label_emb
        x = self.input_proj(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, cond_emb)
        x = self.final_layer(x, cond_emb)
        return x.permute(0, 2, 1)
