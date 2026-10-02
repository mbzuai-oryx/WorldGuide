# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, List, Tuple, Optional, Union, Dict

import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from trainer.distributed import sequence_model_parallel_all_gather
from .modules.activation_layers import get_activation_layer
from .modules.norm_layers import get_norm_layer
from .modules.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, VisionProjection
from .modules.attention import parallel_attention
from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .modules.mlp_layers import MLP, MLPEmbedder, FinalLayer, LinearWarpforSingle
from .modules.modulate_layers import ModulateDiT, modulate, apply_gate
from .modules.token_refiner import SingleTokenRefiner

from trainer.models.hyvideo.utils.communications import all_gather
from trainer.models.hyvideo.utils.infer_utils import torch_compile_wrapper
from trainer.models.hyvideo.models.text_encoders.byT5 import ByT5Mapper

from trainer.distributed.parallel_state import (get_sp_parallel_rank,
                                                  get_sp_world_size)
from trainer.models.prope.camera_rope import prope_qkv
from trainer.configs.models.dits import HunyuanVideoConfig
from trainer.distributed import (get_world_group)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def convpadd(tensor, pad_num):
    dim = tensor.dim()
    if dim == 4:
        tensor = tensor.unsqueeze(2)
    if dim == 6:
        tensor = tensor.squeeze(2)

    b, c, f, h, w = tensor.shape
    pad_h = (pad_num - h % pad_num) % pad_num
    pad_w = (pad_num - w % pad_num) % pad_num
    tensor = torch.cat([tensor, torch.zeros(b, c, f, pad_h, w, device=tensor.device, dtype=tensor.dtype)], dim=3)
    tensor = torch.cat([tensor, torch.zeros(b, c, f, h + pad_h, pad_w, device=tensor.device, dtype=tensor.dtype)], dim=4)
    return tensor


def up_fre(f_1, f_2, f_3, u, f_z, scale=False):
    del scale
    _, _, f1, h1, w1 = u.shape
    freqs_i = torch.cat(
        [
            f_1[f_z : f_z + f1].view(f1, 1, 1, -1).expand(f1, h1, w1, -1),
            f_2[:h1].view(1, h1, 1, -1).expand(f1, h1, w1, -1),
            f_3[:w1].view(1, 1, w1, -1).expand(f1, h1, w1, -1),
        ],
        dim=-1,
    ).reshape(f1 * h1 * w1, 1, -1)
    return freqs_i


def upsample_conv3d_weights_auto(conv_small: nn.Conv3d, size: Tuple[int, int, int]):
    with torch.no_grad():
        w = F.interpolate(conv_small.weight.data, size=size, mode="trilinear", align_corners=False)
        big = nn.Conv3d(
            in_channels=conv_small.in_channels,
            out_channels=conv_small.out_channels,
            kernel_size=size,
            stride=size,
            padding=0,
        )
        big.weight.copy_(w)
        if conv_small.bias is not None:
            big.bias = nn.Parameter(conv_small.bias.data.clone())
        else:
            big.bias = None
    return big


def is_blocks(n: str, m) -> bool:
    print('is_blocks', n, flush=True)
    return "double_blocks" in n and str.isdigit(n.split(".")[-1]) or "single_blocks" in n and str.isdigit(n.split(".")[-1])


class MMDoubleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.factory_kwargs = factory_kwargs

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)

        self.txt_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs)

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    @torch_compile_wrapper()
    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # 添加连续的camera pose，通过prope
        img_q_prope, img_k_prope, img_v_prope, apply_fn_o = prope_qkv(
            img_q.permute(0, 2, 1, 3),
            img_k.permute(0, 2, 1, 3),
            img_v.permute(0, 2, 1, 3),
            viewmats=viewmats,
            Ks=Ks,
        )  # [batch, num_heads, seqlen, head_dim]
        img_q_prope = img_q_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]
        img_k_prope = img_k_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]
        img_v_prope = img_v_prope.permute(0, 2, 1, 3) # [batch, seqlen, num_heads, head_dim]

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        txt_q_before = txt_q.clone()

        # attn_mode = 'flash' if is_flash else self.attn_mode
        attn_mode = 'torch_causal'     # for ar model, the default mode is flex_causal
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

        attn_prope = parallel_attention(
            (img_q_prope, txt_q),
            (img_k_prope, txt_k),
            (img_v_prope, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        img_attn_prope, _ = attn_prope[:, :img_q_prope.shape[1]].contiguous(), attn[:, img_q_prope.shape[1]:].contiguous()
        img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
        img_attn_prope = apply_fn_o(img_attn_prope) # [batch, num_heads, seqlen, head_dim]
        img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")

        img = img + apply_gate(self.img_attn_proj(img_attn) + self.img_attn_prope_proj(img_attn_prope), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return img, txt


class MMSingleStreamBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
    ) -> torch.Tensor:
        """Forward pass for the single stream block."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        txt_mod_shift, txt_mod_scale, txt_mod_gate = self.modulation(vec_txt).chunk(3, dim=-1)

        x_pre_norm = self.pre_norm(x)
        img_token, txt_token = x_pre_norm[:, :-txt_len, :], x_pre_norm[:, -txt_len:, :]

        img_mod = modulate(img_token, shift=mod_shift, scale=mod_scale)
        txt_mod = modulate(txt_token, shift=txt_mod_shift, scale=txt_mod_scale)
        x_mod = torch.cat([img_mod, txt_mod], dim=1)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)
        
        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))
        img_output, txt_output = output[:, :-txt_len, :], output[:, -txt_len:, :]

        img_output = apply_gate(img_output, gate=mod_gate)
        txt_output = apply_gate(txt_output, gate=txt_mod_gate)
        gate_output = torch.cat([img_output, txt_output], dim=1)

        return x + gate_output

class ARHunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    _fsdp_shard_conditions = HunyuanVideoConfig()._fsdp_shard_conditions

    _supports_gradient_checkpointing = True
    _allowed_missing_memory_compression_keys = (
        "patch_embedding.weight",
        "patch_embedding.bias",
        "patch_embedding_2x.weight",
        "patch_embedding_2x.bias",
        "patch_embedding_4x.weight",
        "patch_embedding_4x.bias",
        "patch_embedding_8x.weight",
        "patch_embedding_8x.bias",
        "patch_embedding_16x.weight",
        "patch_embedding_16x.bias",
        "patch_embedding_2x_f.weight",
        "patch_embedding_2x_f.bias",
    )

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str = None,
        ideal_task: str = None,
    ):
        super().__init__()
        factory_kwargs = {}

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False
            )

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels, concat_condition=concat_condition, **factory_kwargs
        )
        # Exact Approach-1 memory compression modules for mode2 path.
        self.patch_embedding = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
            **factory_kwargs,
        )
        with torch.no_grad():
            self.patch_embedding.weight.copy_(self.img_in.proj.weight[:, : self.in_channels])
            if self.img_in.proj.bias is not None:
                self.patch_embedding.bias.copy_(self.img_in.proj.bias)
            else:
                nn.init.zeros_(self.patch_embedding.bias)
        self.patch_embedding_2x = upsample_conv3d_weights_auto(self.patch_embedding, (1, 4, 4))
        self.patch_embedding_4x = upsample_conv3d_weights_auto(self.patch_embedding, (1, 8, 8))
        self.patch_embedding_8x = upsample_conv3d_weights_auto(self.patch_embedding, (1, 16, 16))
        self.patch_embedding_16x = upsample_conv3d_weights_auto(self.patch_embedding, (1, 32, 32))
        self.patch_embedding_2x_f = nn.Conv3d(
            self.patch_embedding.in_channels,
            self.patch_embedding.in_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
            **factory_kwargs,
        )
        self.d = self.hidden_size // self.heads_num
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
            ],
            dim=1,
        )
        self.memory_debug = os.getenv("MEMORY_DEBUG", "0") == "1"

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )
        self.vector_in = (
            MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            ) if self.text_pool_type is not None else None
        )
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )


        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # STA
        if attn_param is None:
            self.attn_param = { 
                # STA
                'win_size': [[3, 3, 3]],
                'win_type': "fixed",
                'win_ratio': 10,
                'tile_size': [6, 8, 8],
                # SSTA
                'ssta_topk': 64,
                'ssta_threshold': 0.0,
                'ssta_lambda': 0.7,
                'ssta_sampling_type': 'importance',
                'ssta_adaptive_pool': None,
                # flex-block-attn:
                "attn_sparse_type": 'ssta',
                "attn_pad_type": 'zero',
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param

        if attn_mode == 'flex-block-attn':
            self.register_to_config(attn_param=self.attn_param)

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size) 
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None

        self.gradient_checkpointing = False

    @classmethod
    def get_allowed_missing_memory_compression_keys(cls) -> set[str]:
        return set(cls._allowed_missing_memory_compression_keys)

    def initialize_memory_compression_modules(self, missing_keys: set[str] | None = None) -> None:
        if missing_keys is None:
            missing_keys = self.get_allowed_missing_memory_compression_keys()
        else:
            missing_keys = set(missing_keys)

        def _xavier_conv(conv: nn.Conv3d):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        def _copy_from_img_in_if_compatible() -> bool:
            src = self.img_in.proj
            dst = self.patch_embedding
            if src.weight.shape[0] != dst.weight.shape[0]:
                return False
            if src.weight.shape[2:] != dst.weight.shape[2:]:
                return False
            if src.weight.shape[1] < self.in_channels:
                return False
            with torch.no_grad():
                dst.weight.copy_(src.weight[:, : self.in_channels])
                if src.bias is not None and dst.bias is not None and src.bias.shape == dst.bias.shape:
                    dst.bias.copy_(src.bias)
                elif dst.bias is not None:
                    nn.init.zeros_(dst.bias)
            return True

        need_patch_base = any(k.startswith("patch_embedding.") for k in missing_keys)
        if need_patch_base:
            copied = _copy_from_img_in_if_compatible()
            if not copied:
                _xavier_conv(self.patch_embedding)

        def _safe_init_multiscale(conv: nn.Conv3d):
            if conv is self.patch_embedding_2x_f:
                nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
                if conv.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(conv.bias, -bound, bound)
            else:
                _xavier_conv(conv)

        if any(k.startswith("patch_embedding_2x.") for k in missing_keys):
            _safe_init_multiscale(self.patch_embedding_2x)
        if any(k.startswith("patch_embedding_4x.") for k in missing_keys):
            _safe_init_multiscale(self.patch_embedding_4x)
        if any(k.startswith("patch_embedding_8x.") for k in missing_keys):
            _safe_init_multiscale(self.patch_embedding_8x)
        if any(k.startswith("patch_embedding_16x.") for k in missing_keys):
            _safe_init_multiscale(self.patch_embedding_16x)
        if any(k.startswith("patch_embedding_2x_f.") for k in missing_keys):
            _safe_init_multiscale(self.patch_embedding_2x_f)

    def load_hunyuan_state_dict(self, model_path):
        load_key = "module"
        bare_model = "unknown"

        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=True
            )

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )

        result = self.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            logger.info("[load.py] Missing keys when loading state_dict:")
            for key in result.missing_keys:
                logger.info(f"[load.py] Missing key: {key}")
        if result.unexpected_keys:
            logger.info("[load.py] Unexpected keys when loading state_dict:")
            for key in result.unexpected_keys:
                logger.info(f"[load.py] Unexpected key: {key}")
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")

        return result

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], byt5_txt_i[~byt5_text_mask_i], txt_i[~text_mask_i]], dim=0
                    )
                reorder_mask_i = torch.cat(
                    [byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i], 
                     byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def add_discrete_action_parameters(self):
        self.action_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu")
        )
        nn.init.zeros_(self.action_in.mlp[2].weight)
        # 如果你想初始化偏置为0，执行以下操作：
        if self.action_in.mlp[2].bias is not None:
            nn.init.zeros_(self.action_in.mlp[2].bias)

        # prope参数,零初始化
        for block in self.double_blocks:
            block.img_attn_prope_proj = nn.Linear(block.hidden_size, block.hidden_size, bias=block.qkv_bias, **block.factory_kwargs)
            nn.init.zeros_(block.img_attn_prope_proj.weight)

            # 如果你想初始化偏置为0，执行以下操作：
            if block.img_attn_prope_proj.bias is not None:
                nn.init.zeros_(block.img_attn_prope_proj.bias)

    def _compress_condition_memory_tokens(
        self,
        cond_latents: torch.Tensor,
        valid_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        def _hist_idx(idx, embed, pad=0, compressed=False):
            return {
                "mode": "idx",
                "idx": idx,
                "embed": embed,
                "pad": pad,
                "compressed": compressed,
            }

        def _hist_slice(
            start,
            end,
            embed,
            pad=0,
            compressed=False,
            fallback_le=None,
            fallback_idx=None,
        ):
            return {
                "mode": "slice",
                "start": start,
                "end": end,
                "embed": embed,
                "pad": pad,
                "compressed": compressed,
                "fallback_le": fallback_le,
                "fallback_idx": fallback_idx,
            }

        history_tiers = [
            (
                2 + 4,
                [
                    _hist_idx(0, "patch_embedding"),
                    _hist_slice(
                        1,
                        -1,
                        "patch_embedding_2x",
                        pad=4,
                        compressed=True,
                        fallback_le=2,
                        fallback_idx=-1,
                    ),
                    _hist_idx(-1, "patch_embedding"),
                ],
            ),
            (
                2 + 4 + 16,
                [
                    _hist_idx(0, "patch_embedding"),
                    _hist_slice(
                        1,
                        -5,
                        "patch_embedding_4x",
                        pad=8,
                        compressed=True,
                        fallback_le=6,
                        fallback_idx=-5,
                    ),
                    _hist_slice(-5, -3, "patch_embedding_2x", pad=4, compressed=True),
                    _hist_slice(-3, None, "patch_embedding"),
                ],
            ),
            (
                2 + 4 + 16 + 64,
                [
                    _hist_idx(0, "patch_embedding"),
                    _hist_slice(
                        1,
                        -21,
                        "patch_embedding_8x",
                        pad=16,
                        compressed=True,
                        fallback_le=22,
                        fallback_idx=-21,
                    ),
                    _hist_slice(-21, -5, "patch_embedding_4x", pad=8, compressed=True),
                    _hist_slice(-5, -3, "patch_embedding_2x", pad=4, compressed=True),
                    _hist_slice(-3, None, "patch_embedding"),
                ],
            ),
            (
                2 + 4 + 16 + 64 + 256,
                [
                    _hist_idx(0, "patch_embedding_2x", pad=4),
                    _hist_slice(
                        1,
                        -85,
                        "patch_embedding_16x",
                        pad=32,
                        compressed=True,
                        fallback_le=86,
                        fallback_idx=-85,
                    ),
                    _hist_slice(-85, -21, "patch_embedding_8x", pad=16, compressed=True),
                    _hist_slice(-21, -5, "patch_embedding_4x", pad=8, compressed=True),
                    _hist_slice(-5, -3, "patch_embedding_2x", pad=4, compressed=True),
                    _hist_slice(-3, None, "patch_embedding"),
                ],
            ),
            (
                2 + 4 + 16 + 64 + 256 + 1024,
                [
                    _hist_idx(0, "patch_embedding_2x", pad=4),
                    _hist_slice(
                        1,
                        -341,
                        "_composed_32x",
                        compressed=True,
                        fallback_le=342,
                        fallback_idx=-341,
                    ),
                    _hist_slice(-341, -85, "patch_embedding_16x", pad=32, compressed=True),
                    _hist_slice(-85, -21, "patch_embedding_8x", pad=16, compressed=True),
                    _hist_slice(-21, -5, "patch_embedding_4x", pad=8, compressed=True),
                    _hist_slice(-5, -3, "patch_embedding_2x", pad=4, compressed=True),
                    _hist_slice(-3, None, "patch_embedding"),
                ],
            ),
        ]

        def _select_history_specs(f_hist):
            for max_frames, specs in history_tiers:
                if f_hist <= max_frames:
                    return specs
            raise ValueError(
                f"Unsupported history length: {f_hist}. "
                "Current memory compression supports up to 1366 historical frames."
            )

        def _take_history_frames(u1, spec):
            f_zero = u1.shape[2]

            if spec["mode"] == "idx":
                return u1[:, :, spec["idx"]].unsqueeze(2)

            if spec["fallback_le"] is not None and f_zero <= spec["fallback_le"]:
                return u1[:, :, spec["fallback_idx"]].unsqueeze(2)

            return u1[:, :, spec["start"] : spec["end"]]

        def _apply_history_embedding(frames, embed_name, pad):
            if embed_name == "_composed_32x":
                return self.patch_embedding_16x(
                    convpadd(
                        self.patch_embedding_2x_f(convpadd(frames, 4)),
                        32,
                    )
                )

            if pad > 0:
                frames = convpadd(frames, pad)

            return getattr(self, embed_name)(frames)

        def _make_freq(f_1, f_2, f_3, emb, offset, compressed):
            if compressed:
                return up_fre(f_1, f_2, f_3, emb, offset, True)
            return up_fre(f_1, f_2, f_3, emb, offset)

        def _compress_history(u1, f_1, f_2, f_3):
            f_hist = u1.shape[2]

            if f_hist <= 0:
                raise ValueError("flag=True requires at least one historical frame.")

            specs = _select_history_specs(f_hist)

            embedded_segments = []
            freqs_parts = []
            offset = 0
            debug_ratios = []

            for spec in specs:
                frames = _take_history_frames(u1, spec)
                emb = _apply_history_embedding(
                    frames,
                    spec["embed"],
                    spec["pad"],
                )
                freqs_parts.append(
                    _make_freq(
                        f_1,
                        f_2,
                        f_3,
                        emb,
                        offset,
                        spec["compressed"],
                    )
                )
                embedded_segments.append(emb)
                offset += emb.shape[2]
                if spec["embed"] == "_composed_32x":
                    debug_ratios.append(32)
                elif "16x" in spec["embed"]:
                    debug_ratios.append(16)
                elif "8x" in spec["embed"]:
                    debug_ratios.append(8)
                elif "4x" in spec["embed"]:
                    debug_ratios.append(4)
                elif "2x" in spec["embed"]:
                    debug_ratios.append(2)
                else:
                    debug_ratios.append(1)

            freqs_i = torch.cat(freqs_parts, dim=0)

            u1 = torch.cat(
                [emb.flatten(2).transpose(1, 2) for emb in embedded_segments],
                dim=1,
            )
            f_z = offset
            return u1, freqs_i, f_z, debug_ratios

        x_pack = []
        freqs_pack = []
        latest_ratios = []
        for sample_idx in range(cond_latents.shape[0]):
            u = cond_latents[sample_idx : sample_idx + 1, :, :valid_t]
            u1 = u[:, :, :-1]
            u2 = u[:, :, -1:]

            f_1 = rope_params(1024, self.d - 4 * (self.d // 6)).to(u.device)
            f_2 = rope_params(1024, 2 * (self.d // 6)).to(u.device)
            f_3 = rope_params(1024, 2 * (self.d // 6)).to(u.device)

            u1, freqs_i, f_z, debug_ratios = _compress_history(u1, f_1, f_2, f_3)
            u2 = self.patch_embedding(u2)
            freqs_i = torch.cat(
                [
                    freqs_i,
                    up_fre(f_1, f_2, f_3, u2, f_z),
                ],
                dim=0,
            )
            u2 = u2.flatten(2).transpose(1, 2)
            u = torch.cat([u1, u2], dim=1)
            x_pack.append(u)
            freqs_pack.append(freqs_i)
            latest_ratios = debug_ratios + [1]

        x = torch.cat(x_pack, dim=0)
        self.freqs = freqs_pack[0]
        return x, self.freqs, latest_ratios

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        action: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=torch.bfloat16
            )

        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        memory_tokens = None
        memory_mode = (
            extra_kwargs.get("origami_memory_mode", "mode1")
            if extra_kwargs is not None
            else "mode1"
        )
        if memory_mode == "mode2" and x.shape[1] >= 2 * self.in_channels + 1:
            cond_latents = x[:, self.in_channels : 2 * self.in_channels]
            cond_mask = x[:, 2 * self.in_channels : 2 * self.in_channels + 1]
            frame_mask = cond_mask[:, 0, :, 0, 0] > 0.5
            valid_t = int(frame_mask[0].sum().item())
            assert bool(torch.all(frame_mask.sum(dim=1) == valid_t)), "All samples must share valid_t in mode2."
            if dist.is_available() and dist.is_initialized():
                valid_t_tensor = torch.tensor(
                    [valid_t], device=cond_latents.device, dtype=torch.int64
                )
                dist.all_reduce(valid_t_tensor, op=dist.ReduceOp.MIN)
                valid_t = int(valid_t_tensor.item())
            if valid_t > 1:
                memory_nonzero = cond_latents[:, :, : valid_t - 1].abs().sum().item()
                local_has_memory = 1 if (math.isfinite(memory_nonzero) and memory_nonzero > 0.0) else 0
                if dist.is_available() and dist.is_initialized():
                    has_mem_tensor = torch.tensor(
                        [local_has_memory], device=cond_latents.device, dtype=torch.int64
                    )
                    dist.all_reduce(has_mem_tensor, op=dist.ReduceOp.MIN)
                    global_has_memory = int(has_mem_tensor.item())
                else:
                    global_has_memory = local_has_memory
                if self.memory_debug:
                    print(
                        f"[memory-debug] model-forward memory_nonzero={memory_nonzero:.6f}, valid_t={valid_t}, global_has_memory={global_has_memory}",
                        flush=True,
                    )
                if global_has_memory > 0:
                    memory_tokens, _, debug_ratios = self._compress_condition_memory_tokens(
                        cond_latents,
                        valid_t,
                    )
                    if self.memory_debug:
                        print(
                            f"[memory-debug] entered tiered compression branch, ratios(old->new)={debug_ratios}",
                            flush=True,
                        )
                    assert any(r > 1 for r in debug_ratios[:-1]), "Older memory frames were not compressed."
                    assert debug_ratios[-1] == 1, "Latest frame should be least compressed (1x)."
                else:
                    valid_t = 1
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        self.attn_param['thw'] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)

        sp_world_size = get_sp_world_size()
        rank_in_sp_group = get_sp_parallel_rank()
        if sp_world_size > 1:
            sp_size = sp_world_size
            sp_rank = rank_in_sp_group
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Too short context length for SP {sp_size}'
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
            Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]
            action = torch.chunk(action, sp_size, dim=0)[sp_rank]
            t = torch.chunk(t, sp_size, dim=0)[sp_rank]

        # Prepare modulation vectors
        vec_txt = self.time_in(timestep_txt) # 为了txt prompt单独计算
        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        vec = vec + self.action_in(action) # 添加离散的action

        # Embed text tokens
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb

        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = self.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

        if self.vision_in is not None and (vision_states is not None or memory_tokens is not None):
            if vision_states is not None:
                extra_encoder_hidden_states = self.vision_in(vision_states)
            else:
                extra_encoder_hidden_states = memory_tokens.new_zeros(
                    memory_tokens.shape[0], 0, memory_tokens.shape[-1]
                )
            if memory_tokens is not None:
                extra_encoder_hidden_states = torch.cat(
                    [memory_tokens.to(extra_encoder_hidden_states.dtype), extra_encoder_hidden_states],
                    dim=1,
                )
            # If t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
            if vision_states is not None and mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
                # Set vision tokens to zero to mitigate potential block mask error in SSTA
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]),
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                )
            # Ensure valid tokens precede padding tokens
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2 * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

            txt, text_mask = self.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )

        # print('text embedding shape:', txt.shape, text_mask.shape)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # mask the txt tokens based on the text_mask
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)

        # Pass through double-stream blocks
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and self.attn_param["win_ratio"] > 0
                and (
                    (index + 1) % self.attn_param["win_ratio"] == 0
                    or (index + 1) == len(self.double_blocks)
                )
            )
            self.attn_param["layer-name"] = f"double_block_{index+1}"

            img, txt = block(
                img=img,
                txt=txt,
                vec_txt=vec_txt,
                vec=vec,
                freqs_cis=freqs_cis,
                text_mask=None,      # we have masked txt tokens already, set None here
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
                viewmats=viewmats,
                Ks=Ks,
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Merge image and text for single-stream blocks
        x = torch.cat((img, txt), 1)
        features_list = [] if output_features else None
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                force_full_attn = (
                    self.attn_mode in ["flex-block-attn"]
                    and self.attn_param["win_type"] == "hybrid"
                    and self.attn_param["win_ratio"] > 0
                    and (
                        (index + 1) % self.attn_param["win_ratio"] == 0
                        or (index + 1) == len(self.single_blocks)
                    )
                )
                self.attn_param["layer-name"] = f"single_block_{index+1}"
                x = block(
                    x=x,
                    vec_txt=vec_txt,
                    vec=vec,
                    txt_len=txt_seq_len,
                    freqs_cis=(freqs_cos, freqs_sin),
                    text_mask=text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                )

                if output_features and index % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])
        img = x[:, :img_seq_len, ...]

        # Final Layer
        img = self.final_layer(img, vec)
        if sp_world_size > 1:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
            if sp_world_size > 1:
                features_list = sequence_model_parallel_all_gather(features_list, dim=2)
        else:
            features_list = None
        return (img, features_list)

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        self.attn_mode = attn_mode
        for block in self.double_blocks:
            block.attn_mode = attn_mode 
        for block in self.single_blocks:
            block.attn_mode = attn_mode
