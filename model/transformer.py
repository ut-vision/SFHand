# adapted from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath

try:
    from flash_attn.modules.mha import MHA as FlashMHA
    from flash_attn.modules.mlp import Mlp as FlashMlp
except:
    print('First pip install flash-attn')
    
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
            layer_idx: int = None
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        else:
            # self.attn = FlashMHA(d_model, n_head, cross_attn=False, bias=True, dropout=attn_drop, use_flash_attn=True)
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, qkv_proj_bias=True,
                                 out_proj_bias=True, dropout=attn_drop, use_flash_attn=True,
                                 layer_idx=layer_idx)
            
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_flash_attn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("drop1", nn.Dropout(drop)),
                ("c_proj", nn.Linear(mlp_width, d_model)),
                ("drop2", nn.Dropout(drop)),
            ]))
        else:
            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, inference_params=None):
        if not self.use_flash_attn:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
        else:
            x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x), inference_params=inference_params)))
        x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_flash_attn=use_flash_attn, layer_idx=layer_idx)
            for layer_idx in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, inference_params=None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask, inference_params=inference_params)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            num_frames: int = 1,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            output_dim: int = None,
            patch_dropout: float = 0.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            ln_pre: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.width = width
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=width, bias=not ln_pre)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        assert num_frames >= 1
        self.num_frames = num_frames
        if num_frames > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(num_frames, width))

        assert not (patch_dropout > 0. and drop_rate > 0.)
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        if ln_pre:
            self.ln_pre = norm_layer(width)
        else:
            self.ln_pre = nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        # TODO: compare the two styles
        # Mimicking timm's initialization
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError('Unknown parameters named {}'.format(n)) 
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection, std=self.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def get_pos_embedding(self, x: torch.Tensor, curr_frames: int):
        # x: [b,c,f,h,w]
        cls_embed = self.positional_embedding[0, :].unsqueeze(0)
        if self.num_frames == curr_frames:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
            tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
        else:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(curr_frames, 1)
            new_temporal_embed = F.interpolate(self.temporal_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.temporal_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            tile_temporal_embed = torch.nn.Parameter(new_temporal_embed).to(self.temporal_embedding.device)
            tile_temporal_embed = tile_temporal_embed.repeat_interleave(self.patches_per_frame, 0)
            
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
        return total_pos_embed

    def forward(self, x: torch.Tensor, return_dense=False, inference_params=None):
        x = x.to(torch.float16)
        curr_frames = x.size(2)
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:

                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])

                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)

                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)    # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)

        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x, inference_params=inference_params)

        if return_dense:
            if self.global_average_pool:
                return self.ln_post(x)
            else:
                return self.ln_post(x[:,1:])
        
        if self.global_average_pool:
            x_pooling = x.mean(dim=1)
        else:
            x_pooling = x[:, 0]

        x_pooling = self.ln_post(x_pooling)

        if self.image_projection is not None:
            x = x @ self.image_projection
            x_pooling = x_pooling @ self.image_projection
        return x_pooling,x

class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            causal_mask: float = True,
            flash_attn: bool = False,
            flash_mlp: bool = False,
            fused_bias_fc: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)
        if output_dim is None:
            self.text_projection = None
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.causal_mask = causal_mask

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)
            # trunc_normal_(self.text_projection, std=0.001)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, cast_dtype=None):
        #with torch.no_grad():
        if cast_dtype is None:
            cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask if self.causal_mask else None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(cast_dtype)

        return x

class CrossAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, query_len, num_layers=4, num_heads=8):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_embed = nn.Parameter(torch.rand(query_len, hidden_dim))

    def forward(self, memory):
        # memory: (bs, token, hidden_dim)
        bs = memory.size(0)
        query_embed = self.query_embed.unsqueeze(1).repeat(1, bs, 1)  # (seq, bs, hidden_dim)
        memory = memory.permute(1, 0, 2)  # (token, bs, hidden_dim)
        decoded = self.decoder(tgt=query_embed, memory=memory)  # (seq, bs, hidden_dim)
        decoded = decoded.permute(1, 0, 2)  # (bs, seq, hidden_dim)
        return decoded


class ResidualTemporalBlock(nn.Module):
    """
    A complete Transformer block with residual connection and KV cache,
    inspired by the structure of ResidualAttentionBlock, designed for streaming processing.
    """

    def __init__(
            self,
            embed_dim: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = 1e-5,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()

        # --- First sub-block: Self-attention with cache ---
        self.ln_1 = norm_layer(embed_dim)

        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        # For efficiency, merge the projection of Q,K,V into a single linear layer
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ls_1 = LayerScale(embed_dim, ls_init_value)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # KV Cache for streaming
        self.register_buffer('k_cache', None, persistent=False)
        self.register_buffer('v_cache', None, persistent=False)

        # --- Second sub-block: MLP ---
        self.ln_2 = norm_layer(embed_dim)
        mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, mlp_width)),
            ("gelu", act_layer()),
            ("drop1", nn.Dropout(drop)),
            ("c_proj", nn.Linear(mlp_width, embed_dim)),
            ("drop2", nn.Dropout(drop)),
        ]))
        self.ls_2 = LayerScale(embed_dim, ls_init_value)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def clear_cache(self):
        """Clear the KV cache. Called when processing new data stream."""
        self.k_cache = None
        self.v_cache = None
        return self

    def _attention_with_cache(self, x: torch.Tensor, memorize=True, memo_params=None):
        """Perform self-attention calculation and process the KV cache."""
        batch_size, seq_len, d_model = x.shape

        q, k_new, v_new = self.in_proj(x).chunk(3, dim=-1)

        if self.k_cache is not None:
            k = torch.cat([k_new, self.k_cache], dim=1)
            v = torch.cat([v_new, self.v_cache], dim=1)
        else:
            k = k_new
            v = v_new

        '''
        **Comment this line if use KV cache**
        '''
        # k, v = k_new, v_new
        if memorize:
            self.k_cache = k.detach()
            self.v_cache = v.detach()
        total_seq_len = k.shape[1]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, total_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, total_seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if memo_params is not None and 'roi_mask' in memo_params:
            roi_mask = memo_params['roi_mask']  # (bs, video_token, 1)
            
            bs, video_tok, _ = roi_mask.shape
            _, total_tok, _ = k_new.shape
            
            # Generate attn_bias, considering the case where k has memory
            # k's shape is (bs, total_seq_len, d_model), where total_seq_len = memory_len + seq_len
            # We need to generate (bs, n_head, seq_len, total_seq_len) attn_bias
            
            # Create attn_bias, only apply ROI enhancement to the video_token part
            pad_bias = torch.zeros(bs, total_seq_len - seq_len + total_tok - video_tok, device=roi_mask.device, dtype=roi_mask.dtype)
            attn_bias = torch.cat([roi_mask.squeeze(-1), pad_bias], dim=1)
            attn_bias = 1e-3 * F.tanh(memo_params['roi_scale']) * attn_bias[:, None, None, :]
            attn_bias = attn_bias - attn_bias.mean(dim=-1, keepdim=True)
        else:
            attn_bias = None

        context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)

    def forward(self, x: torch.Tensor, memorize=True, memo_params=None):
        # First residual block: Attention
        x = x + self.drop_path1(self.ls_1(self._attention_with_cache(self.ln_1(x), memorize, memo_params)))
        # Second residual block: MLP
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x

class HandFeatureProjector(nn.Module):
    def __init__(self, in_dim=162, out_dim=512, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, out_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # or use max/attention pooling

    def forward(self, hand_feats, valid):
        # hand_feats: (bs, seq, maxh, 162)
        # valid: (bs, seq, 1)
        bs, seq, maxh, feat_dim = hand_feats.shape
        idx = torch.arange(maxh, device=hand_feats.device).view(1, 1, maxh)
        mask = (idx < valid).float()  # (bs, seq, maxh)
        mask_expanded = mask.unsqueeze(-1)
        valid_feats = hand_feats * mask_expanded  # (bs, seq, maxh, 162)

        # Merge seq and maxh into a sequence
        feats = valid_feats.view(bs, seq * maxh, feat_dim)  # (bs, seq*maxh, 162)
        mask_flat = mask.view(bs, seq * maxh)  # (bs, seq*maxh)

        # Project to high dimension
        feats_proj = self.input_proj(feats)  # (bs, seq*maxh, out_dim)

        # Generate attention mask (True for padding)
        attn_mask = (mask_flat == 0)  # (bs, seq*maxh)

        # Transformer aggregation
        feats_encoded = self.transformer(feats_proj, src_key_padding_mask=attn_mask)  # (bs, seq*maxh, out_dim)

        # Only pool the valid tokens
        feats_encoded = feats_encoded * mask_flat.unsqueeze(-1)
        valid_count = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        feat_final = feats_encoded.sum(dim=1) / valid_count  # (bs, out_dim)

        return feat_final
