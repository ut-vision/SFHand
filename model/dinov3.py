import clip
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .transformer import (TextTransformer, CrossAttentionDecoder, ResidualTemporalBlock, HandFeatureProjector)
from einops import rearrange
import torch
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel
from .clip import remap_keys_from_open_clip_to_vit, MultiTaskHead

def DINOV3_VITB16(
    config,
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=768,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = AutoModel.from_pretrained(
    "facebook/dinov3-vitb16-pretrain-lvd1689m", 
    device_map="auto", 
    )

    text_model = TextTransformer(context_length=77, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    decoder = CrossAttentionDecoder(hidden_dim=project_embed_dim, query_len=config.max_hand)
    hf_proj = HandFeatureProjector(in_dim=162, out_dim=project_embed_dim)
    mlp = MultiTaskHead(hidden_dim=project_embed_dim, seq_len=config.pred_clip_length)
    adapter = ResidualTemporalBlock(embed_dim=project_embed_dim, n_head=8,  drop_path=0.1)
    model = DINOV3(embed_dim=project_embed_dim, vision_model=vision_model,
                 decoder=decoder, mlp=mlp, tempo_adapter=adapter, projector=hf_proj,
                 text_model=text_model, freeze_temperature=freeze_temperature,ckpt_path=config.lavila_path)
    
    print("=> loading openai model")
    clip_model, preprocess = clip.load("ViT-B/16", device='cpu')
    remapped_state_dict = remap_keys_from_open_clip_to_vit(
        clip_model.state_dict(),
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

    return model

class DINOV3(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 decoder: nn.Module,
                 mlp: nn.Module,
                 tempo_adapter: nn.Module,
                 projector: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 ckpt_path=None,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model
        self.adapter = tempo_adapter
        self.projector = projector
        self.decoder = decoder
        self.mlp = mlp

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.grid_size = self.visual.grid_size
        self.roi_scale = nn.Parameter(torch.zeros([]))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)

        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None
        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_visual(self, image):
        return self.encode_image(image)

    def encode_image(self, image, inference_params=None):

        outputs = self.visual(image, inference_params=inference_params)
        x_pooling, x = outputs.pooler_output, outputs.last_hidden_state
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x_pooling,x

    def encode_text(self, text, cast_dtype=None):
        if len(text.shape) > 2:
            text = text.squeeze()
        x = self.textual(text)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x
    
    def get_representations(self, image, text):
        image_embed, video_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)
        hand_placeholder = text_embed
        fused_embed = torch.cat([video_embed], dim=1)
        # fused_embed = self.adapter(fused_embed, memorize=False)
        # fused_embed = torch.cat([fused_embed, text_embed.unsqueeze(1)], dim=1)

        # decoded_embed = self.decoder(fused_embed)
        return fused_embed


    def get_roi_mask(self, infer_boxes):
        # infer_boxes: (b, seq, max_hand, 4) cx,cy,w,h
        # image_embed: (b, 1+ grid_size[0] * grid_size[1], embed_dim)
        # grid_size: (h, w)
        # return roi_tokens: (b, seq, max_hand, embed_dim)
        
        # 只使用第一帧的预测结果
        rois = infer_boxes[:, 0, :, :]  # (b, max_hand, 4)
        batch_size, max_hand, _ = rois.shape
        h_grid, w_grid = self.grid_size

        # 创建patch坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h_grid, device=infer_boxes.device, dtype=torch.float32),
            torch.arange(w_grid, device=infer_boxes.device, dtype=torch.float32),
            indexing='ij'
        )
        # 归一化坐标到0-1范围
        y_coords = y_coords / h_grid  # (h_grid, w_grid)
        x_coords = x_coords / w_grid  # (w_grid, w_grid)
        
        # 扩展坐标到batch和hand维度
        x_coords_expanded = x_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, max_hand, -1, -1)  # (b, max_hand, h_grid, w_grid)
        y_coords_expanded = y_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, max_hand, -1, -1)  # (b, max_hand, h_grid, w_grid)
        
        # 扩展rois到空间维度
        cx = rois[:, :, 0:1].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)  # (b, max_hand, h_grid, w_grid)
        cy = rois[:, :, 1:2].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)  # (b, max_hand, h_grid, w_grid)
        w = rois[:, :, 2:3].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)   # (b, max_hand, h_grid, w_grid)
        h = rois[:, :, 3:4].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)   # (b, max_hand, h_grid, w_grid)
        
        # 计算每个patch是否在bbox内
        in_bbox = (
            (x_coords_expanded >= (cx - w/2)) & 
            (x_coords_expanded <= (cx + w/2)) & 
            (y_coords_expanded >= (cy - h/2)) & 
            (y_coords_expanded <= (cy + h/2))
        )  # (b, max_hand, h_grid, w_grid)
        
        masks = rearrange(in_bbox, "bs maxhands gh gw -> bs maxhands (gh gw)")
        
        # 对masks在maxhands维度上取或，即任意一个为1即为1
        masks = masks.max(dim=1, keepdim=True)[0].permute(0, 2, 1)  # (bs, gh*gw, 1)
        masks = masks.float()
        
        cls_pad = torch.zeros(masks.size(0), 1, 1, device=masks.device, dtype=masks.dtype)
        roi_mask = torch.cat([cls_pad, masks], dim=1)
        return roi_mask

    @torch.no_grad()
    def inference(self, image, text, hand_stat, valid, inference_params=None):
        image_embed, video_embed = self.encode_image(image, inference_params)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)
        hand_embed = self.projector(hand_stat, valid)
        fused_embed = torch.cat([video_embed, hand_embed.unsqueeze(1)], dim=1)
        fused_embed = self.adapter(fused_embed, memorize=False)
        fused_embed = torch.cat([fused_embed, text_embed.unsqueeze(1)], dim=1)

        decoded_embed = self.decoder(fused_embed)
        output = self.mlp(decoded_embed)
        return output

    def forward(self,image, future, text, hand_stat, valid, eval_mode=False, roi_boxes=None, inference_params=None):
        image_embed, video_embed = self.encode_image(image, inference_params)
        # video_embed = self.adapter(video_embed)
        # print(image_embed.shape, _.shape)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)
        # return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()
        hand_embed = self.projector(hand_stat, valid)
        # video_embed, text_embed = F.normalize(video_embed, dim=-1), F.normalize(text_embed, dim=-1)
        # hand_embed = F.normalize(hand_embed, dim=-1)
        if eval_mode:
            roi_boxes = self.inference(image, text, hand_stat, valid)['pred_boxes']
            roi_mask = self.get_roi_mask(roi_boxes)
            memo_params = {
                'roi_mask': roi_mask,
                'roi_scale': self.roi_scale
            }
        elif roi_boxes is not None:
            roi_mask = self.get_roi_mask(roi_boxes)
            memo_params = {
                'roi_mask': roi_mask,
                'roi_scale': self.roi_scale
            }
        else:
            memo_params = None

        fused_embed = torch.cat([video_embed, hand_embed.unsqueeze(1)], dim=1)
        fused_embed = self.adapter(fused_embed, memo_params=memo_params)
        fused_embed = torch.cat([fused_embed, text_embed.unsqueeze(1)], dim=1)

        # decoded_embed = self.decoder(video_embed, text_embed.unsqueeze(1))
        decoded_embed = self.decoder(fused_embed)
        output = self.mlp(decoded_embed)
        # for k, v in output.items():
        #     print(k, v.shape)
        output['video_embed'] = video_embed
        output['text_embed'] = text_embed
        output['hand_embed'] = hand_embed
        output['logit_scale'] = self.logit_scale.exp()
        return output