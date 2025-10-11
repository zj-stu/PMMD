from functools import partial
from models_stage_2.vit import VisionTransformer, interpolate_pos_embed
from models_stage_2.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models_stage_2 import box_ops
from tools.multilabel_metrics_for_SAMM import get_multi_label
from timm.models.layers import trunc_normal_

import json
import os
import math


class CSRA_head(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_init=1.0, input_type='image'):
        super().__init__()
        self.num_classes = num_classes
        self.input_type = input_type 
        
        if input_type == 'image':
            self.conv_attention = nn.ModuleList([
                nn.Conv2d(feature_dim, 1, kernel_size=1) for _ in range(num_classes)
            ])
        else:
            self.seq_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 4),
                    nn.ReLU(),
                    nn.Linear(feature_dim // 4, 1)
                ) for _ in range(num_classes)
            ])
        
        self.fc_global = nn.Linear(feature_dim, num_classes)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
    
    def forward(self, patch_tokens, class_token):
        B, seq_len, D = patch_tokens.shape
        S_global = self.fc_global(class_token)
        
        if self.input_type == 'image':
            return self._forward_image(patch_tokens, S_global, B, seq_len, D)
        else:
            return self._forward_text(patch_tokens, S_global, B, seq_len, D)
    
    def _forward_image(self, patch_tokens, S_global, B, seq_len, D):
        h = w = int(math.sqrt(seq_len))
        
        if h * w != seq_len:
            target_size = (h + 1) ** 2
            pad_size = target_size - seq_len
            padding = torch.zeros(B, pad_size, D, device=patch_tokens.device)
            patch_tokens_padded = torch.cat([patch_tokens, padding], dim=1)
            h = w = h + 1
            seq_len = target_size
        else:
            patch_tokens_padded = patch_tokens
        
        spatial_features = patch_tokens_padded.view(B, h, w, D).permute(0, 3, 1, 2)
        
        S_attn = []
        for c in range(self.num_classes):
            attn_map = torch.sigmoid(self.conv_attention[c](spatial_features))
            weighted_feature = attn_map * spatial_features
            pooled = torch.mean(weighted_feature, dim=(2, 3))
            S_attn.append(pooled)
        
        S_attn = torch.stack(S_attn, dim=1)
        S_attn = torch.mean(S_attn, dim=2)
        
        S_final = S_global + self.lambda_param * S_attn
        return S_final
    
    def _forward_text(self, patch_tokens, S_global, B, seq_len, D):
        S_attn = []
        for c in range(self.num_classes):
            attn_scores = self.seq_attention[c](patch_tokens)
            attn_weights = torch.sigmoid(attn_scores).squeeze(-1)
            
            weighted_features = patch_tokens * attn_weights.unsqueeze(-1)
            pooled_feature = torch.mean(weighted_features, dim=1)
            class_score = torch.mean(pooled_feature, dim=1)
            S_attn.append(class_score)
        
        S_attn = torch.stack(S_attn, dim=1)
        
        S_final = S_global + self.lambda_param * S_attn
        return S_final


def KL_divergence(p, q, epsilon=1e-8):
    q = q + epsilon
    kl_div = p * torch.log(p / q)
    return kl_div.sum()


def L_i2t(V, T):
    p = F.softmax(V, dim=-1)
    q = F.softmax(T, dim=-1)
    return KL_divergence(p, q)


def L_t2i(V, T):
    p = F.softmax(T, dim=-1)
    q = F.softmax(V, dim=-1)
    return KL_divergence(p, q)


def L_itm(V, T):
    L_i2t_loss = L_i2t(V, T)
    L_t2i_loss = L_t2i(V, T)
    return L_i2t_loss + L_t2i_loss


def generate_gaussian_heatmap(cx, cy, w, h, image_size=256):
    """Generate Gaussian heatmap from bounding box coordinates."""
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


class GaussianEnhancementModule(nn.Module):
    """Gaussian Enhancement Module (GEM) for processing spatial attention maps."""
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


class PMMD_2(nn.Module):
    def __init__(self, 
                 args=None, 
                 config=None,               
                 text_encoder=None,
                 tokenizer=None,
                 init_deit=True):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForTokenClassification.from_pretrained(
            text_encoder, 
            config=bert_config, 
            label_smoothing=config['label_smoothing']
        )

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=3)
        
        self.fch = CSRA_head(num_classes=2, feature_dim=text_width, lambda_init=0.3, input_type='image')
        self.tch = CSRA_head(num_classes=1, feature_dim=text_width, lambda_init=0.3, input_type='text')

        self.gem = GaussianEnhancementModule(input_size=(256, 256), output_dim=embed_dim)
        self.batch_count = 0
        
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(
            text_encoder, 
            config=bert_config,
            label_smoothing=config['label_smoothing']
        )
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr = nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten = nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        trunc_normal_(self.cls_token_local, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def _enhance_visual_features(self, image_embeds, bbox_coords, image_device):
        """Enhance visual features using Gaussian heatmap attention."""
        cx, cy, w, h = bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2], bbox_coords[:, 3]
        heatmap = generate_gaussian_heatmap(cx, cy, w, h, image_size=(256, 256))
        heatmap = torch.tensor(heatmap).unsqueeze(1).to(image_device)
        
        attention_map = self.gem(heatmap)
        cls_token = image_embeds[:, 0:1, :]
        patch_features = image_embeds[:, 1:, :]
        enhanced_patches = patch_features + attention_map
        enhanced_embeds = torch.cat([cls_token, enhanced_patches], dim=1)
        
        return enhanced_embeds

    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """Compute bounding box loss: L1 & GIoU."""
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text, fake_image_box, fake_text_pos, pred_pos, image_path, W, H, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
            
            multicls_label, real_label_pos = get_multi_label(label, image)
            
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            enhanced_Vc = self._enhance_visual_features(image_embeds, pred_pos, image.device)
            enhanced_Vc_atts = torch.ones(enhanced_Vc.size()[:-1], dtype=torch.long).to(image.device)
            enhanced_Vc_feat = F.normalize(self.vision_proj(enhanced_Vc[:, 0, :]), dim=-1)

            text_output = self.text_encoder.bert(
                text.input_ids, 
                attention_mask=text.attention_mask,
                return_dict=True, 
                mode='text'
            )
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                enhanced_Vc_m = self._enhance_visual_features(image_embeds_m, pred_pos, image.device)
                
                enhanced_Vc_feat_m = F.normalize(self.vision_proj(enhanced_Vc_m[:, 0, :]), dim=-1)
                enhanced_Vc_feat_all = torch.cat([enhanced_Vc_feat_m.t(), self.image_queue.clone().detach()], dim=1)

                text_output_m = self.text_encoder_m.bert(
                    text.input_ids, 
                    attention_mask=text.attention_mask,
                    return_dict=True, 
                    mode='text'
                )
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = enhanced_Vc_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ enhanced_Vc_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets[real_label_pos, real_label_pos] = 1

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets_g2g.fill_diagonal_(1)
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = enhanced_Vc_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ enhanced_Vc_feat_all / self.temp
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            
            sim_i2i = enhanced_Vc_feat @ enhanced_Vc_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets_g2g, dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets_g2g, dim=1).mean()

            loss_MAC = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
            loss_ITM = L_itm(enhanced_Vc_feat, text_feat)
            loss_CSA = loss_MAC + loss_ITM * 0.1

            self._dequeue_and_enqueue(enhanced_Vc_feat_m, text_feat_m)

            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=enhanced_Vc,
                encoder_attention_mask=enhanced_Vc_atts,
                output_attentions=True,
                return_dict=True,
                mode='fusion',
            )
            
            with torch.no_grad():
                bs = image.size(0)

            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
            itm_labels[real_label_pos] = 0
            vl_output = self.itm_head(output_pos.last_hidden_state[:, 0, :])
            loss_BIC = F.cross_entropy(vl_output, itm_labels)

            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone()
            local_feat_padding_mask_text = text_attention_mask_clone == 0

            local_feat_it_cross_attn = enhanced_Vc + self.it_cross_attn(
                query=self.norm_layer_it_cross_atten(enhanced_Vc),
                key=self.norm_layer_it_cross_atten(text_embeds),
                value=self.norm_layer_it_cross_atten(text_embeds),
                key_padding_mask=local_feat_padding_mask_text
            )[0]

            local_feat_aggr = self.aggregator(
                query=self.norm_layer_aggr(cls_tokens_local),
                key=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :]),
                value=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :])
            )[0]
            
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            
            cross_embeds_cls = local_feat_it_cross_attn
            cls_f = self.fch(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
            cls_t = self.tch(output_pos.last_hidden_state[:, 1:, :], output_pos.last_hidden_state[:, 0, :])

            cls = torch.concat((cls_f, cls_t), dim=1)
            loss_MLC = F.binary_cross_entropy_with_logits(cls, multicls_label.type(torch.float))

            token_label = text.attention_mask[:, 1:].clone()
            token_label[token_label == 0] = -100
            token_label[token_label == 1] = 0

            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1

            input_ids = text.input_ids.clone()

            if self.args.token_momentum:
                with torch.no_grad():
                    logits_m = self.text_encoder_m(
                        input_ids,
                        attention_mask=text.attention_mask,
                        encoder_hidden_states=enhanced_Vc_m,
                        encoder_attention_mask=enhanced_Vc_atts,
                        return_dict=True,
                        return_logits=True,
                    )
                token_cls_output = self.text_encoder(
                    input_ids,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=enhanced_Vc,
                    encoder_attention_mask=enhanced_Vc_atts,
                    return_dict=True,
                    labels=token_label,
                    soft_labels=F.softmax(logits_m.view(-1, 2), dim=-1),
                    alpha=alpha
                )
            else:
                token_cls_output = self.text_encoder(
                    input_ids,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=enhanced_Vc,
                    encoder_attention_mask=enhanced_Vc_atts,
                    return_dict=True,
                    labels=token_label,
                )

            loss_TMG = token_cls_output.loss

            return loss_CSA, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC

        else:
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(
                text.input_ids, 
                attention_mask=text.attention_mask,
                return_dict=True, 
                mode='text'
            )
            text_embeds = text_output.last_hidden_state

            enhanced_Vc = self._enhance_visual_features(image_embeds, pred_pos, image.device)
            enhanced_Vc_atts = torch.ones(enhanced_Vc.size()[:-1], dtype=torch.long).to(image.device)
            enhanced_Vc_feat = F.normalize(self.vision_proj(enhanced_Vc[:, 0, :]), dim=-1)

            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=enhanced_Vc,
                encoder_attention_mask=enhanced_Vc_atts,
                return_dict=True,
                mode='fusion',
            )
            
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone()
            local_feat_padding_mask_text = text_attention_mask_clone == 0

            local_feat_it_cross_attn = enhanced_Vc + self.it_cross_attn(
                query=self.norm_layer_it_cross_atten(enhanced_Vc),
                key=self.norm_layer_it_cross_atten(text_embeds),
                value=self.norm_layer_it_cross_atten(text_embeds),
                key_padding_mask=local_feat_padding_mask_text
            )[0]

            local_feat_aggr = self.aggregator(
                query=self.norm_layer_aggr(cls_tokens_local),
                key=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :]),
                value=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :])
            )[0]
            
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()

            logits_real_fake = self.itm_head(output_pos.last_hidden_state[:, 0, :])
            
            cross_embeds_cls = local_feat_it_cross_attn
            cls_f = self.fch(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
            cls_t = self.tch(output_pos.last_hidden_state[:, 1:, :], output_pos.last_hidden_state[:, 0, :])
            logits_multicls = torch.concat((cls_f, cls_t), dim=1)

            logits_multicls = self.cls_head(output_pos.last_hidden_state[:, 0, :])

            input_ids = text.input_ids.clone()
            logits_tok = self.text_encoder(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )
            
            return logits_real_fake, logits_multicls, output_coord, logits_tok

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors."""
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output