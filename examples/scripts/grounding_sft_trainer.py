from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trl import SFTTrainer  # base TRL trainer

# ------------------------------------------------------------
# Geometry utilities: IoU/GIoU + Hungarian
# ------------------------------------------------------------
def _box_iou_giou(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor):
    # pred: (P,4), tgt: (G,4) in absolute xyxy
    tl = torch.maximum(pred_xyxy[:, None, :2], tgt_xyxy[None, :, :2])
    br = torch.minimum(pred_xyxy[:, None, 2:], tgt_xyxy[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    area_g = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0) * (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=0)

    union = area_p[:, None] + area_g[None, :] - inter + 1e-6
    iou = inter / union

    enc_tl = torch.minimum(pred_xyxy[:, None, :2], tgt_xyxy[None, :, :2])
    enc_br = torch.maximum(pred_xyxy[:, None, 2:], tgt_xyxy[None, :, 2:])
    enc_wh = (enc_br - enc_tl).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1] + 1e-6

    giou = iou - (enc_area - union) / enc_area
    return iou, giou

def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        return cost.new_empty((0,), dtype=torch.long), cost.new_empty((0,), dtype=torch.long)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        return cost.new_tensor(r, dtype=torch.long), cost.new_tensor(c, dtype=torch.long)
    except Exception:
        # Greedy fallback
        P, G = cost.shape
        rows = list(range(P)); cols = list(range(G))
        rr, cc = [], []
        work = cost.clone()
        while rows and cols:
            sub = work[rows][:, cols]
            idx = int(sub.argmin())
            i = idx // len(cols); j = idx % len(cols)
            rr.append(rows.pop(i)); cc.append(cols.pop(j))
        return cost.new_tensor(rr, dtype=torch.long), cost.new_tensor(cc, dtype=torch.long)

# ------------------------------------------------------------
# Config + simple box head
# ------------------------------------------------------------
@dataclass
class DinoLossCfg:
    # Matching cost (DETR/DINO): lambda1 * L1 + lambda_g * (1 - GIoU)
    match_l1: float = 5.0
    match_giou: float = 2.0
    # Regression loss after matching
    loss_l1: float = 5.0
    loss_giou: float = 2.0
    # Weight of the auxiliary bbox loss vs CE
    lambda_bbox: float = 1.0
    # Clamp predicted boxes to image plane
    clamp_to_image: bool = True

class BBoxHead(nn.Module):
    """Project hidden states at anchor tokens -> normalized boxes in [0,1]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        # hs: (N, H), returns (N, 4) in [0,1]
        return torch.sigmoid(self.proj(hs))


# ------------------------------------------------------------
# Trainer that adds DINO-style loss to SFT loss
# ------------------------------------------------------------
class GroundingSFTTrainer(SFTTrainer):
    def __init__(self, *args, bbox_head: Optional[nn.Module] = None, dino_cfg: Optional[DinoLossCfg] = None, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if bbox_head is None:
            if hidden_size is None:
                raise ValueError("Model config missing hidden_size; please pass bbox_head explicitly.")
            bbox_head = BBoxHead(hidden_size)
        self.bbox_head = bbox_head
        self.dino_cfg = dino_cfg or DinoLossCfg()

    # ---- helpers -------------------------------------------------
    def _gather_anchor_states(self, last_hidden: torch.Tensor, anchor_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        last_hidden: (B, T, H), anchor_idx: (B, A) with -1 padding.
        Returns list of tensors (P_i, H) per sample.
        """
        B, T, H = last_hidden.shape
        per_states: List[torch.Tensor] = []
        for i in range(B):
            idx = anchor_idx[i]
            idx = idx[idx >= 0]
            if idx.numel() == 0:
                per_states.append(last_hidden.new_zeros((0, H)))
            else:
                per_states.append(last_hidden[i, idx])  # (P_i, H)
        return per_states

    def _per_sample_dino_loss(
        self,
        pred_norm_xyxy: torch.Tensor,   # (P, 4) in [0,1]
        gt_abs_xyxy: torch.Tensor,      # (G, 4) absolute pixels
        hw: torch.Tensor,                # (2,) -> (H, W)
        cfg: DinoLossCfg
    ) -> torch.Tensor:
        if pred_norm_xyxy.numel() == 0 or gt_abs_xyxy.numel() == 0:
            return pred_norm_xyxy.new_zeros(())
        H, W = hw[0].item(), hw[1].item()

        # normalize to absolute, enforce x1<=x2, y1<=y2
        p = pred_norm_xyxy
        x1 = torch.minimum(p[:, 0], p[:, 2]) * W
        x2 = torch.maximum(p[:, 0], p[:, 2]) * W
        y1 = torch.minimum(p[:, 1], p[:, 3]) * H
        y2 = torch.maximum(p[:, 1], p[:, 3]) * H
        pred_abs = torch.stack([x1, y1, x2, y2], dim=-1)

        if cfg.clamp_to_image:
            pred_abs[:, 0::2] = pred_abs[:, 0::2].clamp(0, W)  # x
            pred_abs[:, 1::2] = pred_abs[:, 1::2].clamp(0, H)  # y

        # matching cost
        _, giou = _box_iou_giou(pred_abs, gt_abs_xyxy)
        l1 = (pred_abs[:, None, :] - gt_abs_xyxy[None, :, :]).abs().sum(-1)  # (P, G)
        cost = cfg.match_l1 * l1 + cfg.match_giou * (1 - giou.clamp(-1, 1))
        r, c = _hungarian(cost)
        if r.numel() == 0:
            return pred_norm_xyxy.new_zeros(())

        # regression losses on matches
        l1_m = (pred_abs[r] - gt_abs_xyxy[c]).abs().sum(-1)
        _, giou_full = _box_iou_giou(pred_abs[r], gt_abs_xyxy[c])
        giou_m = giou_full.diag().clamp(-1, 1)
        loss = cfg.loss_l1 * l1_m.mean() + cfg.loss_giou * (1 - giou_m).mean()
        loss = loss / max(1, gt_abs_xyxy.shape[0])  # normalize by #GT (as in DETR)
        return loss

    # ---- your compute_loss with the aux loss blended in ----------
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute CE loss (via super) + DINO-style bbox loss on selected samples.
        Keeps your entropy/accuracy logging intact.
        """
        mode = "train" if self.model.training else "eval"

        # keep labels for accuracy logging
        labels = inputs["labels"]

        # ensure we get hidden states for the box head
        inputs["use_cache"] = False
        inputs["output_hidden_states"] = True

        # ---- 1) standard SFT CE loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # ---- 2) optional entropy metric (unchanged from your code)
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # ---- 3) token accuracy metric (unchanged)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)
                if self.aux_loss_enabled:
                    aux_loss_val = getattr(outputs, "aux_loss", torch.tensor(0.0, device=loss.device))
                    aux_loss_val = self.accelerator.gather_for_metrics(aux_loss_val).mean().item()
                    self._metrics[mode]["aux_loss"].append(aux_loss_val)

        # ---- 4) DINO-style bbox loss (APPLIED ONLY IF FIELDS PRESENT)
        bbox_loss = torch.tensor(0.0, device=loss.device)
        if (
            "box_anchor_indices" in inputs
            and "gt_boxes" in inputs
            and "image_hw" in inputs
            and hasattr(outputs, "hidden_states")
            and outputs.hidden_states is not None
        ):
            last_hidden = outputs.hidden_states[-1]      # (B, T, H)
            anchor_idx  = inputs["box_anchor_indices"]   # (B, A) with -1 padding
            img_hw      = inputs["image_hw"]             # (B, 2)
            weights     = inputs.get("bbox_sample_weights", None)  # (B,) optional

            per_states = self._gather_anchor_states(last_hidden, anchor_idx)  # list[(P_i, H)]
            per_pred_norm = [self.bbox_head(hs) for hs in per_states]         # list[(P_i, 4)]

            # per-sample DINO loss
            per_losses = []
            B = last_hidden.size(0)
            for i in range(B):
                P_i = per_pred_norm[i]
                G_i = inputs["gt_boxes"][i]  # tensor or list
                if isinstance(G_i, list):
                    G_i = torch.tensor(G_i, device=P_i.device, dtype=P_i.dtype)
                else:
                    G_i = G_i.to(P_i.device, dtype=P_i.dtype)
                per_losses.append(self._per_sample_dino_loss(P_i, G_i, img_hw[i], self.dino_cfg))

            per_losses = torch.stack(per_losses) if per_losses else torch.zeros((), device=loss.device)
            if per_losses.ndim == 0:
                per_losses = per_losses[None]  # (1,)

            if weights is not None:
                weights = weights.to(per_losses.device, dtype=per_losses.dtype)
                denom = torch.clamp(weights.sum(), min=1.0)
                bbox_loss = (weights * per_losses).sum() / denom
            else:
                bbox_loss = per_losses.mean()

            # expose for your logger
            outputs.aux_loss = bbox_loss.detach()

            print("Dino Style BBox Loss : ", bbox_loss)
            # blend with CE loss
            loss = loss + self.dino_cfg.lambda_bbox * bbox_loss

        return (loss, outputs) if return_outputs else loss
