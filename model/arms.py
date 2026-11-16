"""Experimental arms for the pinned SpikeYOLO-s upstream source.

The public adapter takes an explicit upstream checkpoint path. Model
construction and loss code come from the experiments described in the paper.
TWIN is a process-wide switch, so select the arm in a fresh process or call
set_twin(False) before using any spiking arm in the same process.
"""
import copy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import yolo_spikformer as ysf
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox

ARMS = ('B', 'STAL', 'STAL_SD', 'P2', 'TWIN')
STAL_ARMS = ('STAL', 'STAL_SD')
STAL_S_MIN, STAL_S_REF = 8.0, 16.0
P2_HEAD_STRIDES = (8.0, 16.0, 32.0, 4.0)
SD_DELTA = 1


# --------------------------------------------------------------------------- anchors
def make_anchors_sub(feats, strides, sub, grid_cell_offset=0.5):
    """make_anchors with a sub-cell factor k per level: a (k*h, k*w) map yields points at (i+0.5)/k in stride units."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        k = int(sub[i])
        sx = (torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset) / k
        sy = (torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset) / k
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


# --------------------------------------------------------------------------- signed DFL head (STAL_SD)
class SDSpikeDetect(ysf.SpikeDetect):
    """SpikeDetect with signed DFL bins (convert_to_sd); decoded widths and heights are kept non-negative."""
    sd_delta = SD_DELTA

    def forward(self, x):
        out = super().forward(x)
        if self.training:
            return out
        (out if self.export else out[0])[:, 2:4].clamp_(min=0)  # xywh: a box whose sides cross decodes as empty
        return out


def convert_to_sd(model, delta=SD_DELTA):
    """Shift the DFL range by delta strides: bin k now means distance k - delta.

    The head regresses LTRB distances from the anchor and DFL bounds them to [0, 15] strides, so an anchor outside a
    tiny box (STAL candidates sit less than one stride outside) can only predict a box that contains the anchor.
    With bins meaning -delta..15-delta that box is exactly representable. Per side, output channel k of the last
    regression conv takes old bin k - delta, the delta new low bins start with zero weights and a -10 bias, and the
    integral weights become -delta..15-delta (still integers): initial predictions stay those of the pretrained head
    except for the dropped top bins (old 16-delta..15).
    """
    head = model.model[-1]
    assert type(head) is ysf.SpikeDetect, 'convert a native SpikeDetect head exactly once'
    n = head.reg_max
    with torch.no_grad():
        for branch in head.cv2:
            conv = branch[-1].conv
            w, b = conv.weight.view(4, n, -1), conv.bias.view(4, n)
            w[:, delta:] = w[:, :n - delta].clone()
            w[:, :delta] = 0.0
            b[:, delta:] = b[:, :n - delta].clone()
            b[:, :delta] = -10.0
        head.dfl.conv.weight.copy_((torch.arange(n, dtype=torch.float) - delta).view(1, n, 1, 1))
    head.__class__ = SDSpikeDetect
    head.sd_delta = int(delta)
    return model


# --------------------------------------------------------------------------- P2 head
def _routed(module, index, source):
    module.i, module.f, module.type = index, source, type(module).__name__
    module.np = sum(q.numel() for q in module.parameters())
    return module


def convert_to_p2(model, seed=0):
    """Dense stride-4 branch with a 4th detection level.

    Nodes 0-25 are unchanged and nothing feeds back into the old neck. Node 19 (stride 8, 128 ch) is upsampled x2,
    concatenated with backbone node 2 (stride 4, 64 ch), projected 192->64 by MS_StandardConv(1x1) and fused by
    MS_AllConvBlock(64, 4, 7). The detection head keeps its three old levels and gains a stride-4 level appended last
    (strides 8, 16, 32, 4), because SpikeDetect sizes the class/box branches of every level from ch[0]; putting the
    64-channel P2 first would change the widths of the pretrained branches. New modules are drawn from their own
    generator seed (global RNG untouched); new class biases use the author's stride prior for stride 4; new BN layers
    take the model's BN eps/momentum. Old state keys are unchanged except the head prefix model.26 -> model.31.
    """
    from ultralytics.nn import tasks
    old_head = model.model[-1]
    assert type(old_head) is ysf.SpikeDetect and len(model.model) == 27 and old_head.nl == 3, 'P2 expects the B model'
    assert old_head.cv2[0][0].conv.in_channels == 128
    ref = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 404)
        projection = ysf.MS_StandardConv(192, 64, 1, 1)
        fusion = ysf.MS_AllConvBlock(64, 4, 7)
        fresh = ysf.SpikeDetect(old_head.nc, (128, 256, 512, 64))
        fresh.stride = torch.tensor(P2_HEAD_STRIDES)
        fresh.bias_init()
    head = old_head
    head.cv2.append(fresh.cv2[3])
    head.cv3.append(fresh.cv3[3])
    head.nl = 4
    head.stride = fresh.stride.clone()
    for new in (projection, fusion, head.cv2[3], head.cv3[3]):
        for m in new.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = ref.eps, ref.momentum
    nodes = list(model.model[:26]) + [
        _routed(nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'), 26, 19),
        _routed(tasks.Concat(2), 27, [26, 2]),
        _routed(projection, 28, -1),
        _routed(fusion, 29, -1),
        _routed(nn.Identity(), 30, -1),
        _routed(head, 31, [19, 22, 25, 30])]
    model.model = nn.Sequential(*nodes).to(device)
    model.save = sorted(set(model.save) | {2, 26, 30})
    model.yaml = copy.deepcopy(model.yaml)
    model.yaml['head'] = model.yaml['head'][:-1] + [
        [19, 1, 'nn.Upsample', [None, '(1,2,2)', 'nearest']],
        [[26, 2], 1, 'Concat', [2]],
        [-1, 1, 'MS_StandardConv', [128, 1, 1]],
        [-1, 1, 'MS_AllConvBlock', [4, 7]],
        [-1, 1, 'nn.Identity', []],
        [[19, 22, 25, 30], 1, 'SpikeDetect', ['nc']]]
    model.stride = head.stride
    head.shape, head.anchors, head.strides = None, torch.empty(0), torch.empty(0)
    return model


# --------------------------------------------------------------------------- assigners
def select_candidates_lowmem(xy_centers, gt_bboxes, eps=1e-9, chunk=128):
    """Same result as vendor select_candidates_in_gts (min of the 4 deltas > eps) without the (b*G, A, 4) tensor."""
    bs, n_boxes, _ = gt_bboxes.shape
    na = xy_centers.shape[0]
    flat = gt_bboxes.reshape(-1, 4)
    out = torch.empty((flat.shape[0], na), dtype=gt_bboxes.dtype, device=gt_bboxes.device)
    x, y = xy_centers[:, 0][None], xy_centers[:, 1][None]
    for s in range(0, flat.shape[0], chunk):
        g = flat[s:s + chunk]
        d = torch.minimum(torch.minimum(x - g[:, 0:1], y - g[:, 1:2]), torch.minimum(g[:, 2:3] - x, g[:, 3:4] - y))
        out[s:s + chunk] = d.gt(eps).to(out.dtype)
    return out.view(bs, n_boxes, na)


class SprintAssigner(TaskAlignedAssigner):
    """Vendor TAL processed image by image with chunked candidate geometry (identical outputs, lower peak memory)."""
    chunk = 128

    def candidate_boxes(self, gt_bboxes):
        return gt_bboxes

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        bs = pd_scores.shape[0]
        if bs == 1:
            return super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        outs = [super(SprintAssigner, self).forward(pd_scores[i:i + 1], pd_bboxes[i:i + 1], anc_points,
                                                    gt_labels[i:i + 1], gt_bboxes[i:i + 1], mask_gt[i:i + 1])
                for i in range(bs)]
        return tuple(torch.cat(t, 0) for t in zip(*outs))

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = select_candidates_lowmem(anc_points, self.candidate_boxes(gt_bboxes), chunk=self.chunk)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps


class STALAssigner(SprintAssigner):
    """Candidate selection on a surrogate box; everything else (IoU, scores, targets) uses the original box."""
    s_min, s_ref = STAL_S_MIN, STAL_S_REF

    def candidate_boxes(self, gt_bboxes):
        x1, y1, x2, y2 = gt_bboxes.unbind(-1)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        w = torch.where(w < self.s_min, torch.full_like(w, self.s_ref), w)
        h = torch.where(h < self.s_min, torch.full_like(h, self.s_ref), h)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


# --------------------------------------------------------------------------- loss
class SprintDetectionLoss(v8DetectionLoss):
    """Vendor v8DetectionLoss; only the image size and anchor generation become sub-cell aware."""

    def __init__(self, model, stal=False):
        super().__init__(model)
        head = model.model[-1]
        self.sub = tuple(int(k) for k in getattr(head, 'sub', (1,) * len(self.stride)))
        assigner = STALAssigner if stal else SprintAssigner
        self.assigner = assigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0] / self.sub[0]
        anchor_points, stride_tensor = make_anchors_sub(feats, self.stride, self.sub, 0.5)

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


def signed_bbox2dist(anchor_points, bbox, reg_max, delta):
    """bbox2dist on the signed range: LTRB distances clamped to [-delta, reg_max - delta - 0.01] strides."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(-delta, reg_max - delta - 0.01)


class SignedBboxLoss(BboxLoss):
    """Vendor BboxLoss with signed DFL targets: distance d is learned on bins d + delta (see convert_to_sd)."""

    def __init__(self, reg_max, delta=SD_DELTA):
        super().__init__(reg_max, use_dfl=True)
        self.delta = delta

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        target_ltrb = signed_bbox2dist(anchor_points, target_bboxes, self.reg_max, self.delta) + self.delta
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
        return loss_iou, loss_dfl.sum() / target_scores_sum


class SDDetectionLoss(SprintDetectionLoss):
    """SprintDetectionLoss for signed DFL bins (STAL_SD): shifted integral, signed targets, non-crossing boxes."""

    def __init__(self, model, stal=True, delta=SD_DELTA):
        super().__init__(model, stal=stal)
        self.delta = delta
        self.bbox_loss = SignedBboxLoss(self.reg_max - 1, delta).to(self.device)
        self.proj = self.proj - delta

    def bbox_decode(self, anchor_points, pred_dist):
        x1y1, x2y2 = super().bbox_decode(anchor_points, pred_dist).chunk(2, -1)
        return torch.cat((x1y1, torch.maximum(x2y2, x1y1)), -1)  # signed sides may cross: keep boxes non-negative


# --------------------------------------------------------------------------- twin
_NATIVE_QUANT4_FORWARD = ysf.MultiSpike4.quant4.__dict__['forward']


def _continuous_forward(ctx, input):
    ctx.save_for_backward(input)
    return torch.clamp(input, min=0, max=4)


def set_twin(enabled):
    """Process-wide switch: integer firing round(clamp(x,0,4)) <-> continuous clamp(x,0,4). Backward is unchanged."""
    ysf.MultiSpike4.quant4.forward = staticmethod(_continuous_forward) if enabled else _NATIVE_QUANT4_FORWARD


def twin_enabled():
    return ysf.MultiSpike4.quant4.__dict__['forward'] is not _NATIVE_QUANT4_FORWARD


# --------------------------------------------------------------------------- model
class SprintDetectionModel(DetectionModel):
    sprint_arm = 'B'

    def init_criterion(self):
        arm = getattr(self, 'sprint_arm', 'B')
        loss = SDDetectionLoss if arm == 'STAL_SD' else SprintDetectionLoss
        return loss(self, stal=arm in STAL_ARMS)


def build_model(arm, weight, nc=8, device='cpu', seed=0):
    """Initialise from the official COCO 23M T1/D4 weights, 749 tensors transferred, 6 fresh class outputs.

    The pinned DetectionModel.__init__ hard-codes .cuda() (tasks.py:241,251), so construction needs a visible GPU;
    the finished model is moved to `device` (the trainer later moves it to its own device).
    """
    assert arm in ARMS, f'unknown arm {arm}'
    set_twin(arm == 'TWIN')
    ckpt = torch.load(Path(weight), map_location='cpu', weights_only=False)
    pre = ckpt.get('ema') if ckpt.get('ema') is not None else ckpt['model']
    model = SprintDetectionModel(copy.deepcopy(pre.yaml), nc=nc, verbose=False)
    sd, msd = pre.float().state_dict(), model.state_dict()
    transfer = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    result = model.load_state_dict(transfer, strict=False)
    assert len(transfer) == 749 and len(result.missing_keys) == 6 and not result.unexpected_keys, \
        (len(transfer), result.missing_keys, result.unexpected_keys)
    if arm == 'P2':
        convert_to_p2(model, seed)
    if arm == 'STAL_SD':
        convert_to_sd(model)
    model.sprint_arm = arm
    return model.to(device)
