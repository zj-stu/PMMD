import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from timm.models.layers import trunc_normal_

from models_stage_1.xvlm import XVLMBase, XVLMPlusBase, VanillaConfig, build_mlp
from models_stage_1.beit2 import PatchEmbed
from models_stage_1 import box_ops
from tools.multilabel_metrics_for_DGM4 import get_multi_label


def generate_gaussian_heatmap(cx, cy, w, h, image_size=256):
    if isinstance(image_size, int):
        H, W = image_size, image_size
    else:
        H, W = image_size
    
    device = cx.device
    original_size = (256, 256)
    W_orig, H_orig = original_size
    
    cx_pixel = cx * W_orig
    cy_pixel = cy * H_orig
    w_pixel = w * W_orig
    h_pixel = h * H_orig
    
    sigma = torch.sqrt((h_pixel / 2) ** 2 + (w_pixel / 2) ** 2)
    
    x = torch.arange(image_size[0], device=device).float()
    y = torch.arange(image_size[1], device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    cx_pixel = cx_pixel.view(-1, 1, 1)
    cy_pixel = cy_pixel.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    
    distance_sq = (xx - cx_pixel) ** 2 + (yy - cy_pixel) ** 2
    heatmap = torch.exp(-distance_sq / (2 * (sigma ** 2) + 1e-8))
    heatmap = torch.clamp(heatmap, 0, 1)
    
    return heatmap


def box_cxcywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred_bbox, target_bbox):
    x1 = torch.max(pred_bbox[:, 0], target_bbox[:, 0])
    y1 = torch.max(pred_bbox[:, 1], target_bbox[:, 1])
    x2 = torch.min(pred_bbox[:, 2], target_bbox[:, 2])
    y2 = torch.min(pred_bbox[:, 3], target_bbox[:, 3])
    
    inter_area = torch.max(x2 - x1, torch.zeros_like(x2)) * torch.max(y2 - y1, torch.zeros_like(y2))
    pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
    target_area = (target_bbox[:, 2] - target_bbox[:, 0]) * (target_bbox[:, 3] - target_bbox[:, 1])
    union_area = pred_area + target_area - inter_area
    
    iou = inter_area / union_area
    return iou


def generate_patch_labels(images, norm_bboxes, patch_size=16):
    B, C, H, W = images.shape
    device = images.device
    num_patches = H // patch_size
    
    cx = norm_bboxes[..., 0] * W
    cy = norm_bboxes[..., 1] * H
    w = norm_bboxes[..., 2] * W
    h = norm_bboxes[..., 3] * H
    
    xmin = torch.clamp(cx - w / 2, 0, W)
    ymin = torch.clamp(cy - h / 2, 0, H)
    xmax = torch.clamp(cx + w / 2, 0, W)
    ymax = torch.clamp(cy + h / 2, 0, H)
    
    grid = torch.stack(torch.meshgrid(
        torch.arange(num_patches, device=device) * patch_size,
        torch.arange(num_patches, device=device) * patch_size,
    ), dim=-1)
    
    blocks = torch.cat([grid, grid + patch_size], dim=-1)
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    boxes = boxes.view(B, -1, 1, 1, 4)
    blocks = blocks.view(1, 1, num_patches, num_patches, 4)
    
    inter_x1 = torch.maximum(blocks[..., 0], boxes[..., 0])
    inter_y1 = torch.maximum(blocks[..., 1], boxes[..., 1])
    inter_x2 = torch.minimum(blocks[..., 2], boxes[..., 2])
    inter_y2 = torch.minimum(blocks[..., 3], boxes[..., 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    patch_labels = (inter_area > 0).any(dim=1).view(B, -1).float()
    
    return patch_labels


class Geometric_enhance(nn.Module):
    def __init__(self, input_size=256, output_dim=768):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(256, 768, kernel_size=1),
        )
    
    def forward(self, heatmap):
        x = self.conv_net(heatmap)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class PatchWiseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, visual_features, patch_labels):
        visual_features = visual_features[:, 1:, :]
        visual_features_norm = F.normalize(visual_features, p=2, dim=-1)
        
        similarity_matrix = torch.bmm(
            visual_features_norm, 
            visual_features_norm.transpose(1, 2)
        ) / self.temperature
        
        target = (patch_labels.unsqueeze(2) == patch_labels.unsqueeze(1)).float()
        
        loss = F.binary_cross_entropy_with_logits(
            similarity_matrix,
            target,
            reduction='none'
        )
        
        mask = torch.eye(similarity_matrix.size(1), dtype=torch.bool, device=similarity_matrix.device)
        loss = loss.masked_fill(mask, 0).mean()
        
        return loss


class DFP(nn.Module):
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6):
        super().__init__()
        
        self.num_queries = num_queries
        self.query_dim = hidden_dim
        
        self.query_modal = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.query_cross = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        
        self.cross_attention_modal = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=12, 
            dropout=0.0, 
            batch_first=True
        )
        self.cross_attention_cross = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=12, 
            dropout=0.0, 
            batch_first=True
        )
        
        self.self_attention_modal = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=12, 
            dropout=0.0, 
            batch_first=True
        )
        self.self_attention_cross = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=12, 
            dropout=0.0, 
            batch_first=True
        )
        
        self.ffn_modal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.ffn_cross = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        trunc_normal_(self.query_modal, std=0.02)
        trunc_normal_(self.query_cross, std=0.02)
        trunc_normal_(self.positional_encoding, std=0.02)
    
    def forward(self, visual_modal, visual_cross):
        bs = visual_modal.size(0)
        query_modal = self.query_modal.expand(bs, -1, -1)
        query_cross = self.query_cross.expand(bs, -1, -1)
        
        query_modal, _ = self.cross_attention_modal(query_modal, visual_modal, visual_modal)
        query_cross, _ = self.cross_attention_cross(query_cross, visual_cross, visual_cross)
        
        query_modal_detached = query_modal.detach()
        query_cross_detached = query_cross.detach()
        
        query_combined_modal = torch.cat((query_modal, query_cross_detached), dim=1)
        query_combined_cross = torch.cat((query_modal_detached, query_cross), dim=1)
        
        query_combined_modal, _ = self.self_attention_modal(
            query_combined_modal, 
            query_combined_modal, 
            query_combined_modal
        )
        query_combined_cross, _ = self.self_attention_cross(
            query_combined_cross, 
            query_combined_cross, 
            query_combined_cross
        )
        
        output_modal = self.ffn_modal(query_combined_modal.squeeze(1))
        output_cross = self.ffn_cross(query_combined_cross.squeeze(1))
        
        return output_modal, output_cross


class PMMD_1(XVLMBase):
    def __init__(self, args=None, config=None, load_vision_params=True, 
                 load_text_params=True, pretraining=True):
        super().__init__(
            config, 
            load_vision_params=load_vision_params, 
            load_text_params=load_text_params,
            use_contrastive_loss=False, 
            use_matching_loss=False, 
            use_mlm_loss=False, 
            use_bbox_loss=True,
            config_text=None, 
            pretraining=pretraining
        )
        
        self.args = args
        self.gem = Geometric_enhance(input_size=(256, 256), output_dim=self.embed_dim)
        self.pwc_loss = PatchWiseContrastiveLoss()
        self.dfp = DFP()
        
        self.bbox_head_intra = build_mlp(input_dim=self.text_width, output_dim=4)
        self.bbox_head_cross_step1 = build_mlp(input_dim=self.text_width, output_dim=4)
        self.bbox_head_cross_step2 = build_mlp(input_dim=self.text_width, output_dim=4)
        self.bbox_head_cross_step3 = build_mlp(input_dim=self.text_width, output_dim=4)
    
    def _enhance_visual_features(self, image_embeds, bbox_coords, image_device):
        cx, cy, w, h = bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2], bbox_coords[:, 3]
        heatmap = generate_gaussian_heatmap(cx, cy, w, h, image_size=(256, 256))
        heatmap = torch.tensor(heatmap).unsqueeze(1).to(image_device)
        
        attention_map = self.gem(heatmap)
        cls_token = image_embeds[:, 0:1, :]
        patch_features = image_embeds[:, 1:, :]
        enhanced_patches = patch_features + attention_map
        enhanced_embeds = torch.cat([cls_token, enhanced_patches], dim=1)
        
        return enhanced_embeds
    
    def _FLM(self, visual_embeds, text_embeds, text_attention_mask, 
                                     cls_token_local, batch_size):
        text_mask_clone = text_attention_mask.clone()
        padding_mask = text_mask_clone == 0
        
        cross_attn_output = visual_embeds + self.it_cross_attn(
            query=self.norm_layer_it_cross_atten(visual_embeds),
            key=self.norm_layer_it_cross_atten(text_embeds),
            value=self.norm_layer_it_cross_atten(text_embeds),
            key_padding_mask=padding_mask
        )[0]
        
        aggregated_features = self.aggregator(
            query=self.norm_layer_aggr(cls_token_local),
            key=self.norm_layer_aggr(cross_attn_output[:, 1:, :]),
            value=self.norm_layer_aggr(cross_attn_output[:, 1:, :])
        )[0]
        
        return aggregated_features
    
    def forward(self, image, image_path, label, text, fake_image_box, fake_text_pos,
                text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None,
                ret_bbox_loss=True, ret_match_loss=False,
                alpha=0, is_train=True):
        
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
            
            multicls_label, real_label_pos = get_multi_label(label, image)
            text_ids = text.input_ids
            text_atts = text.attention_mask
            
            image_embeds, image_atts = self.get_vision_embeds(image)
            visual_features, _ = self.get_vision_embeds_f(image)
            text_embeds = self.get_text_embeds(text_ids, text_atts)
            
            loss_mac, loss_itm = self.get_mac_loss(
                image, image_embeds, text_embeds, real_label_pos,
                idx_to_group_img=None, text_ids=text_ids, text_atts=text_atts,
                temp=self.temp, alpha=0.5
            )
            
            patch_labels = generate_patch_labels(image, fake_image_box)
            loss_pwc = self.pwc_loss(visual_features, patch_labels)
            
            _, feature_dual = self.dfp(visual_features, image_embeds)
            feature_intra = feature_dual[:, 0].squeeze(1)
            bbox_intra = self.bbox_head_intra(feature_intra).sigmoid()
            loss_bbox_intra, loss_giou_intra = self.get_bbox_loss(bbox_intra, fake_image_box)
            
            bbox_cross_step1, _ = self.FLM(image, image_embeds, text_embeds, text_atts)
            loss_bbox_step1, loss_giou_step1 = self.get_bbox_loss(bbox_cross_step1, fake_image_box)
            
            bs = image.size(0)
            enhanced_embeds_step1 = self._enhance_visual_features(
                image_embeds, bbox_cross_step1, image.device
            )
            cls_token_step1 = self.cls_token_local_L.expand(bs, -1, -1)
            aggregated_step1 = self._FLM(
                enhanced_embeds_step1, text_embeds, text.attention_mask, cls_token_step1, bs
            )
            bbox_cross_step2 = self.bbox_head_cross_step2(aggregated_step1.squeeze(1)).sigmoid()
            loss_bbox_step2, loss_giou_step2 = self.get_bbox_loss(bbox_cross_step2, fake_image_box)
            
            enhanced_embeds_step2 = self._enhance_visual_features(
                image_embeds, bbox_cross_step2, image.device
            )
            cls_token_step2 = self.cls_token_local_L_2.expand(bs, -1, -1)
            aggregated_step2 = self._FLM(
                enhanced_embeds_step2, text_embeds, text.attention_mask, cls_token_step2, bs
            )
            bbox_cross_step3 = self.bbox_head_cross_step3(aggregated_step2.squeeze(1)).sigmoid()
            loss_bbox_step3, loss_giou_step3 = self.get_bbox_loss(bbox_cross_step3, fake_image_box)
            
            loss_bbox_total = loss_bbox_intra + loss_bbox_step1 + loss_bbox_step2 + loss_bbox_step3
            loss_giou_total = loss_giou_intra + loss_giou_step1 + loss_giou_step2 + loss_giou_step3
            loss_csa = loss_mac + loss_itm * 0.1
            
            return loss_csa, loss_pwc, loss_bbox_total, loss_giou_total
        
        else:
            multicls_label, real_label_pos = get_multi_label(label, image)
            bs = image.size(0)
            
            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
            itm_labels[real_label_pos] = 0
            
            if ret_bbox_loss:
                image_embeds, image_atts = self.get_vision_embeds(
                    image, image_atts=image_atts, idx_to_group_img=idx_to_group_img
                )
            else:
                image_embeds, image_atts = self.get_vision_embeds(image)
            
            visual_features, _ = self.get_vision_embeds_f(image)
            text_ids = text.input_ids
            text_atts = text.attention_mask
            text_embeds = self.get_text_embeds(text_ids, text_atts)
            
            _, feature_dual = self.dfp(visual_features, image_embeds)
            feature_intra = feature_dual[:, 0].squeeze(1)
            bbox_intra = self.bbox_head_intra(feature_intra).sigmoid()
            
            bbox_cross_step1, _ = self.FLM(image, image_embeds, text_embeds, text_atts)
            enhanced_embeds_step1 = self._enhance_visual_features(
                image_embeds, bbox_cross_step1, image.device
            )
            cls_token_step1 = self.cls_token_local_L.expand(bs, -1, -1)
            aggregated_step1 = self._FLM(
                enhanced_embeds_step1, text_embeds, text.attention_mask, cls_token_step1, bs
            )
            bbox_cross_step2 = self.bbox_head_cross_step2(aggregated_step1.squeeze(1)).sigmoid()
            
            enhanced_embeds_step2 = self._enhance_visual_features(
                image_embeds, bbox_cross_step2, image.device
            )
            cls_token_step2 = self.cls_token_local_L_2.expand(bs, -1, -1)
            aggregated_step2 = self._FLM(
                enhanced_embeds_step2, text_embeds, text.attention_mask, cls_token_step2, bs
            )
            
            aggregated_step2_combined = aggregated_step2.squeeze(1) + feature_intra
            bbox_final = self.bbox_head_cross_step3(aggregated_step2_combined).sigmoid()
            
            return bbox_final


def write_to_dataset(cx_pred, cy_pred, w_pred, h_pred, image_path):
    cx_pred = cx_pred.detach().clone().cpu().numpy().tolist()
    cy_pred = cy_pred.detach().clone().cpu().numpy().tolist()
    w_pred = w_pred.detach().clone().cpu().numpy().tolist()
    h_pred = h_pred.detach().clone().cpu().numpy().tolist()
    
    bbox_data = {
        "image_path": image_path,
        "pred_bboxes": {
            "cx": cx_pred,
            "cy": cy_pred,
            "w": w_pred,
            "h": h_pred
        }
    }
    
    os.makedirs("bbox_results", exist_ok=True)
    
    try:
        if torch.distributed.is_initialized():
            gpu_rank = torch.distributed.get_rank()
        else:
            gpu_rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
    except:
        gpu_rank = 0
    
    filename = f"bbox_results/gpu_{gpu_rank}_batches.json"
    all_data = []
    
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_data = []
    
    all_data.append(bbox_data)
    
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=4)


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )