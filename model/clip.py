import clip
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .transformer import (TextTransformer, VisionTransformer,
                          CrossAttentionDecoder, ResidualTemporalBlock, HandFeatureProjector)
from einops import rearrange
import torch
from collections import OrderedDict

# util functions to convert OpenCLIP-style model keys to ViT-style
def remap_keys_from_open_clip_to_vit(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "logit_scale": "logit_scale",
        "visual.proj": "visual.image_projection",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }

    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.transformer.resblocks.{layer}.{tgt_name}"


    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
             'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    for key in clip_state_dict:
        if key in ["visual.proj", "text_projection", "logit_scale"]:
            continue
        if use_fast_conv1 and key == 'visual.conv1.weight':
            remapped_state_dict['visual.conv1.weight'] = clip_state_dict[key].flatten(1)
        elif key not in key_mapping:
            remapped_state_dict[key] = clip_state_dict[key]
        else:
            if key == 'positional_embedding':
                old_context_length, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                if context_length <= old_context_length:
                    remapped_state_dict[key_mapping[key]] = clip_state_dict[key][:context_length, :]
                else:
                    remapped_state_dict[key_mapping[key]] = torch.cat(
                        (clip_state_dict[key], torch.zeros((context_length - old_context_length, dim), dtype=old_dtype)), dim=0
                    )
            elif key == 'token_embedding.weight':
                old_vocab_size, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                assert vocab_size >= old_vocab_size
                remapped_state_dict[key_mapping[key]] = torch.cat(
                    (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
                )
            else:
                remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict

def CLIP_VITB16(
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
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )

    text_model = TextTransformer(context_length=77, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    decoder = CrossAttentionDecoder(hidden_dim=project_embed_dim, query_len=config.max_hand)
    hf_proj = HandFeatureProjector(in_dim=162, out_dim=project_embed_dim)
    mlp = MultiTaskHead(hidden_dim=project_embed_dim, seq_len=config.pred_clip_length)
    adapter = ResidualTemporalBlock(embed_dim=project_embed_dim, n_head=8,  drop_path=0.1)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model,
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

class CLIP(nn.Module):
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
        self.grid_size = self.visual.grid_size
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

        x_pooling, x = self.visual(image, inference_params=inference_params)
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
    
    def get_representations(self, image, text, proprior=None):
        image_embed, video_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        if proprior is not None:
            hand_feat = self.prop_adapter(proprior)
            valid = torch.ones(hand_feat.shape[0], 1, 1, device=hand_feat.device)
            hand_embed = self.projector(hand_feat[:, None, None], valid)
            fused_embed = torch.cat([video_embed, hand_embed.unsqueeze(1)], dim=1)
        else:
            fused_embed = torch.cat([video_embed], dim=1)

        fused_embed = self.adapter(fused_embed, memorize=False)
        fused_embed = torch.cat([fused_embed, text_embed.unsqueeze(1)], dim=1)

        # decoded_embed = self.decoder(fused_embed)
        return fused_embed


    def get_roi_mask(self, infer_boxes):
        '''
        infer_boxes: (b, seq, max_hand, 4) cx,cy,w,h
        image_embed: (b, 1+ grid_size[0] * grid_size[1], embed_dim)
        grid_size: (h, w)
        return roi_tokens: (b, seq, max_hand, embed_dim)
        '''
        
        # Only use the first frame's prediction result
        rois = infer_boxes[:, 0, :, :]  # (b, max_hand, 4)
        batch_size, max_hand, _ = rois.shape
        h_grid, w_grid = self.grid_size

        # Create patch coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h_grid, device=infer_boxes.device, dtype=torch.float32),
            torch.arange(w_grid, device=infer_boxes.device, dtype=torch.float32),
            indexing='ij'
        )
        # Normalize coordinates to 0-1 range
        y_coords = y_coords / h_grid  # (h_grid, w_grid)
        x_coords = x_coords / w_grid  # (w_grid, w_grid)
        
        # Expand coordinates to batch and hand dimensions
        x_coords_expanded = x_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, max_hand, -1, -1)  # (b, max_hand, h_grid, w_grid)
        y_coords_expanded = y_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, max_hand, -1, -1)  # (b, max_hand, h_grid, w_grid)
        
        # Expand rois to spatial dimensions
        cx = rois[:, :, 0:1].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)  # (b, max_hand, h_grid, w_grid)
        cy = rois[:, :, 1:2].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)  # (b, max_hand, h_grid, w_grid)
        w = rois[:, :, 2:3].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)   # (b, max_hand, h_grid, w_grid)
        h = rois[:, :, 3:4].unsqueeze(-1).expand(-1, -1, h_grid, w_grid)   # (b, max_hand, h_grid, w_grid)
        
        # Calculate whether each patch is in the bbox
        in_bbox = (
            (x_coords_expanded >= (cx - w/2)) & 
            (x_coords_expanded <= (cx + w/2)) & 
            (y_coords_expanded >= (cy - h/2)) & 
            (y_coords_expanded <= (cy + h/2))
        )  # (b, max_hand, h_grid, w_grid)
        
        masks = rearrange(in_bbox, "bs maxhands gh gw -> bs maxhands (gh gw)")
        
        # Take or on the maxhands dimension, i.e., any one is 1 is 1
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
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)
        hand_embed = self.projector(hand_stat, valid)
            
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

        decoded_embed = self.decoder(fused_embed)
        output = self.mlp(decoded_embed)
        output['video_embed'] = video_embed
        output['text_embed'] = text_embed
        output['hand_embed'] = hand_embed
        output['logit_scale'] = self.logit_scale.exp()
        return output


class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim=512, seq_len=8):
        super().__init__()
        self.box_head = nn.Linear(hidden_dim, 4 * seq_len)
        self.class_head = nn.Linear(hidden_dim, 3 * seq_len)
        self.camera_head = nn.Linear(hidden_dim, 3 * seq_len)
        self.orient_head = nn.Linear(hidden_dim, 9 * seq_len)
        self.pose_head = nn.Linear(hidden_dim, 135 * seq_len)
        self.betas_head = nn.Linear(hidden_dim, 10 * seq_len)
        self.seq_len = seq_len

    def forward(self, x):
        box = self.box_head(x).sigmoid()
        cls = self.class_head(x)
        camera_t = self.camera_head(x)
        orient = self.orient_head(x)
        pose = self.pose_head(x)
        betas = self.betas_head(x)
        outputs = {'pred_boxes': box,
                   'pred_logits': cls,
                   'pred_camera_t': camera_t,
                   'pred_global_orient': orient,
                   'pred_hand_pose': pose,
                   'pred_betas': betas}
        for k, v in outputs.items():
            outputs[k] = rearrange(v, "bs q (seq dim) -> bs seq q dim", seq=self.seq_len)
        outputs['pred_logits'] = outputs['pred_logits'].softmax(-1)
        return outputs
