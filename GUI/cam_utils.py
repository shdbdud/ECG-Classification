import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from PIL import Image, ImageDraw

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, img_tensor, rr_tensor, class_idx=1):
        """
        img_tensor: (1,1,H,W)
        rr_tensor : (1,rr_dim)
        class_idx : 1 表示 abnormal
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(img_tensor, rr_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # GAP over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,h,w)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def get_target_layer(model_name, model):
    """
    根据模型名称返回合适的 Grad-CAM 目标层
    """

    if model_name == "Original_DSR_Attn_Proto":
        return model.morph.b3

    if model_name == "Enhanced_DSR_SE":
        return model.b3_conv

    if model_name == "Original_DSR":
        return model.morph.b3

    if model_name == "Pure_CNN":
        return model.conv3

    if model_name == "CNN_LSTM":
        return model.cnn[4]

    if model_name == "Pure_LSTM":
        raise ValueError("Pure_LSTM is not suitable for standard Grad-CAM on 2D feature maps.")

    raise ValueError(f"Unsupported model for CAM: {model_name}")


def overlay_cam_on_image(img_tensor, cam, alpha=0.35):
    """
    img_tensor: (1,1,H,W) or (1,H,W)
    cam: H x W in [0,1]
    return:
        base_rgb: H x W x 3
        overlay : H x W x 3
        heatmap : H x W x 3
    """
    if img_tensor.ndim == 4:
        img_np = img_tensor[0, 0].detach().cpu().numpy()
    elif img_tensor.ndim == 3:
        img_np = img_tensor[0].detach().cpu().numpy()
    else:
        raise ValueError("img_tensor must have shape (1,1,H,W) or (1,H,W)")

    img_np = img_np.astype(np.float32)
    img_np = img_np - img_np.min()
    img_np = img_np / (img_np.max() + 1e-8)

    base_rgb = np.stack([img_np, img_np, img_np], axis=-1)

    heatmap = cm.jet(cam)[..., :3]  # RGBA -> RGB
    overlay = (1 - alpha) * base_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return base_rgb, overlay, heatmap

def cam_to_time_segments(
    cam,
    top_quantile=0.88,
    min_width_ratio=0.05,
    merge_gap_ratio=0.03,
    max_segments=3,
):
    """
    把 2D CAM 压成 1D 时间重要性，再取连续高响应区段
    return:
        segments: [(x1, x2, score), ...]
        profile : 1D importance profile, shape (W,)
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError("cam must be 2D, shape (H, W)")

    profile = cam.mean(axis=0)  # 沿纵轴压缩，保留时间轴
    profile = profile - profile.min()
    profile = profile / (profile.max() + 1e-8)

    W = profile.shape[0]
    thr = float(np.quantile(profile, top_quantile))
    mask = profile >= thr

    min_width = max(4, int(W * min_width_ratio))
    merge_gap = max(2, int(W * merge_gap_ratio))

    raw_segments = []
    start = None

    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif (not flag) and (start is not None):
            end = i - 1
            if end - start + 1 >= min_width:
                score = float(profile[start:end + 1].mean())
                raw_segments.append((start, end, score))
            start = None

    if start is not None:
        end = W - 1
        if end - start + 1 >= min_width:
            score = float(profile[start:end + 1].mean())
            raw_segments.append((start, end, score))

    # 如果阈值太高，至少保留一个主峰区段
    if len(raw_segments) == 0:
        peak = int(np.argmax(profile))
        half = max(6, int(W * 0.05))
        x1 = max(0, peak - half)
        x2 = min(W - 1, peak + half)
        score = float(profile[x1:x2 + 1].mean())
        raw_segments.append((x1, x2, score))

    # 合并相邻过近的区段
    raw_segments = sorted(raw_segments, key=lambda t: t[0])
    merged = []

    for seg in raw_segments:
        if not merged:
            merged.append(list(seg))
            continue

        px1, px2, ps = merged[-1]
        cx1, cx2, cs = seg

        if cx1 - px2 <= merge_gap:
            nx1 = px1
            nx2 = max(px2, cx2)
            ns = max(ps, cs)
            merged[-1] = [nx1, nx2, ns]
        else:
            merged.append(list(seg))

    # 先按分数取前 max_segments，再按横坐标排序
    merged = [(int(x1), int(x2), float(s)) for x1, x2, s in merged]
    merged = sorted(merged, key=lambda t: t[2], reverse=True)[:max_segments]
    merged = sorted(merged, key=lambda t: t[0])

    return merged, profile


def draw_segments_on_original_ecg(
    image_pil,
    segments,
    cam_width,
    y_top_ratio=0.72,
    y_bottom_ratio=0.94,
    color=(220, 38, 38),
    alpha=0.20,
    line_width=4,
):
    """
    把 CAM 时间区段映射回原始 ECG 图，并画半透明红色异常区块

    image_pil: 原始上传的 ECG 图 (PIL.Image)
    segments : [(cam_x1, cam_x2, score), ...]
    cam_width: CAM 的宽度，用来做横向比例映射

    return:
        out_img: 画好区块的 RGB PIL.Image
        mapped_segments: [
            {
                "rank": 1,
                "cam_x1": ...,
                "cam_x2": ...,
                "img_x1": ...,
                "img_x2": ...,
                "score": ...
            }, ...
        ]
    """
    base = image_pil.convert("RGBA")
    W, H = base.size

    y1 = max(0, min(H - 1, int(H * y_top_ratio)))
    y2 = max(0, min(H - 1, int(H * y_bottom_ratio)))
    if y2 <= y1:
        y1, y2 = int(H * 0.72), int(H * 0.94)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mapped_segments = []
    cam_width = max(1, int(cam_width))

    fill_rgba = (color[0], color[1], color[2], int(255 * alpha))
    line_rgba = (color[0], color[1], color[2], 255)

    for rank, (sx1, sx2, score) in enumerate(segments, start=1):
        ix1 = int(round((sx1 / max(cam_width - 1, 1)) * (W - 1)))
        ix2 = int(round((sx2 / max(cam_width - 1, 1)) * (W - 1)))

        ix1 = max(0, min(W - 1, ix1))
        ix2 = max(0, min(W - 1, ix2))
        if ix2 < ix1:
            ix1, ix2 = ix2, ix1

        draw.rectangle(
            [ix1, y1, ix2, y2],
            fill=fill_rgba,
            outline=line_rgba,
            width=line_width,
        )

        mapped_segments.append(
            {
                "rank": rank,
                "cam_x1": int(sx1),
                "cam_x2": int(sx2),
                "img_x1": int(ix1),
                "img_x2": int(ix2),
                "score": float(score),
            }
        )

    out_img = Image.alpha_composite(base, overlay).convert("RGB")
    return out_img, mapped_segments
def find_suspicious_region(cam, top_quantile=0.90):
    """
    沿横轴找最可疑区间
    return:
        x1, x2, profile
    """
    profile = cam.mean(axis=0)   # HxW -> W
    thr = np.quantile(profile, top_quantile)
    idx = np.where(profile >= thr)[0]

    if len(idx) == 0:
        return None, None, profile

    x1 = int(idx[0])
    x2 = int(idx[-1])
    return x1, x2, profile


def profile_to_image(profile, height=80):
    """
    把 1D profile 变成一个简单可显示的灰度图（可选）
    """
    profile = np.asarray(profile, dtype=np.float32)
    profile = profile - profile.min()
    profile = profile / (profile.max() + 1e-8)

    canvas = np.ones((height, len(profile)), dtype=np.float32)
    for x, v in enumerate(profile):
        y = int((1 - v) * (height - 1))
        canvas[y:, x] = 0.2
    return canvas

def draw_suspicious_box_on_overlay(overlay, x1, x2, color=(1.0, 1.0, 0.0), thickness=2):
    """
    overlay: H x W x 3, range [0,1]
    x1, x2: suspicious horizontal interval
    color: RGB in [0,1], default yellow
    """
    if overlay is None or x1 is None or x2 is None:
        return overlay

    img = overlay.copy()
    h, w, _ = img.shape

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))

    if x2 < x1:
        x1, x2 = x2, x1

    t = max(1, int(thickness))

    # top
    img[0:t, x1:x2+1, :] = color
    # bottom
    img[h-t:h, x1:x2+1, :] = color
    # left
    img[:, x1:x1+t, :] = color
    # right
    img[:, x2-t+1:x2+1, :] = color

    return img

import numpy as np

from PIL import Image, ImageDraw

def draw_exact_trace_segments_on_original_ecg(
    image_pil,
    segments,
    vis_meta,
    cam_width,
    tube_radius=10,
    color=(220, 38, 38),
    alpha=70,
    line_width=4,
):
    """
    segments: [(sx1, sx2, score), ...]
    vis_meta : preprocess 时保存的几何信息
    cam_width: cam.shape[1]
    """
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    trace_x = np.asarray(vis_meta["trace_x"], dtype=np.float32)
    trace_y = np.asarray(vis_meta["trace_y"], dtype=np.float32)

    n = len(trace_x)
    if n == 0:
        return base.convert("RGB")

    for sx1, sx2, score in segments:
        i1 = int(round((sx1 / max(cam_width - 1, 1)) * (n - 1)))
        i2 = int(round((sx2 / max(cam_width - 1, 1)) * (n - 1)))
        i1 = max(0, min(n - 1, i1))
        i2 = max(0, min(n - 1, i2))
        if i2 < i1:
            i1, i2 = i2, i1

        pts = [(float(trace_x[i]), float(trace_y[i])) for i in range(i1, i2 + 1)]

        # 先画粗线当作高亮带
        if len(pts) >= 2:
            draw.line(pts, fill=(color[0], color[1], color[2], alpha), width=tube_radius * 2)

        # 再画细边界线
        if len(pts) >= 2:
            draw.line(pts, fill=(color[0], color[1], color[2], 255), width=line_width)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def draw_exact_trace_circles_on_original_ecg(
    image_pil,
    segments,
    vis_meta,
    cam_width,
    cam_profile=None,
    radius=18,
    color=(220, 38, 38),
    alpha=60,
    line_width=4,
):
    """
    在原始 ECG 上，用空心圆圈标出最可疑的位置
    segments: [(sx1, sx2, score), ...]
    vis_meta : preprocess_image(return_meta=True) 返回的几何信息
    cam_width: cam.shape[1]
    cam_profile: 1D profile，优先用它在每个 segment 内找真正峰值
    """
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    trace_x = np.asarray(vis_meta["trace_x"], dtype=np.float32)
    trace_y = np.asarray(vis_meta["trace_y"], dtype=np.float32)

    n = len(trace_x)
    if n == 0:
        return base.convert("RGB"), []

    points_info = []

    for rank, (sx1, sx2, score) in enumerate(segments, start=1):
        # 在 segment 内找最强响应点
        if cam_profile is not None:
            local = np.asarray(cam_profile[sx1:sx2 + 1], dtype=np.float32)
            if len(local) > 0:
                peak_cam_x = int(np.argmax(local)) + int(sx1)
            else:
                peak_cam_x = int(round((sx1 + sx2) / 2))
        else:
            peak_cam_x = int(round((sx1 + sx2) / 2))

        idx = int(round((peak_cam_x / max(cam_width - 1, 1)) * (n - 1)))
        idx = max(0, min(n - 1, idx))

        cx = int(round(trace_x[idx]))
        cy = int(round(trace_y[idx]))

        # 半透明填充
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(color[0], color[1], color[2], alpha),
            outline=(color[0], color[1], color[2], 255),
            width=line_width,
        )

        points_info.append(
            {
                "rank": rank,
                "cam_x": int(peak_cam_x),
                "img_x": int(cx),
                "img_y": int(cy),
                "score": float(score),
            }
        )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out, points_info

def draw_exact_trace_circles_on_original_ecg(
    image_pil,
    segments,
    vis_meta,
    cam_width,
    cam_profile=None,
    radius=18,
    color=(220, 38, 38),
    alpha=60,
    line_width=4,
):
    """
    在原始 ECG 上，用空心圆圈标出最可疑的位置
    segments: [(sx1, sx2, score), ...]
    vis_meta : preprocess_image(return_meta=True) 返回的几何信息
    cam_width: cam.shape[1]
    cam_profile: 1D profile，优先用它在每个 segment 内找真正峰值
    """
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    trace_x = np.asarray(vis_meta["trace_x"], dtype=np.float32)
    trace_y = np.asarray(vis_meta["trace_y"], dtype=np.float32)

    n = len(trace_x)
    if n == 0:
        return base.convert("RGB"), []

    points_info = []

    for rank, (sx1, sx2, score) in enumerate(segments, start=1):
        # 在 segment 内找最强响应点
        if cam_profile is not None:
            local = np.asarray(cam_profile[sx1:sx2 + 1], dtype=np.float32)
            if len(local) > 0:
                peak_cam_x = int(np.argmax(local)) + int(sx1)
            else:
                peak_cam_x = int(round((sx1 + sx2) / 2))
        else:
            peak_cam_x = int(round((sx1 + sx2) / 2))

        idx = int(round((peak_cam_x / max(cam_width - 1, 1)) * (n - 1)))
        idx = max(0, min(n - 1, idx))

        cx = int(round(trace_x[idx]))
        cy = int(round(trace_y[idx]))

        # 半透明填充
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(color[0], color[1], color[2], alpha),
            outline=(color[0], color[1], color[2], 255),
            width=line_width,
        )

        points_info.append(
            {
                "rank": rank,
                "cam_x": int(peak_cam_x),
                "img_x": int(cx),
                "img_y": int(cy),
                "score": float(score),
            }
        )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out, points_info
def cam_profile_to_peaks(profile, top_n=4, min_distance_ratio=0.08, min_score=0.35):
    profile = np.asarray(profile, dtype=np.float32)
    profile = profile - profile.min()
    profile = profile / (profile.max() + 1e-8)

    W = len(profile)
    min_dist = max(8, int(W * min_distance_ratio))

    work = profile.copy()
    peaks = []

    for _ in range(top_n):
        x = int(np.argmax(work))
        s = float(work[x])
        if s < min_score:
            break

        peaks.append((x, s))

        x1 = max(0, x - min_dist)
        x2 = min(W, x + min_dist + 1)
        work[x1:x2] = -1.0

    return peaks


def draw_exact_trace_peak_circles_on_original_ecg(
    image_pil,
    peaks,
    vis_meta,
    cam_width,
    radius=18,
    color=(220, 38, 38),
    alpha=60,
    line_width=4,
):
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    trace_x = np.asarray(vis_meta["trace_x"], dtype=np.float32)
    trace_y = np.asarray(vis_meta["trace_y"], dtype=np.float32)

    n = len(trace_x)
    if n == 0:
        return base.convert("RGB"), []

    mapped_points = []

    for rank, (peak_cam_x, score) in enumerate(peaks, start=1):
        idx = int(round((peak_cam_x / max(cam_width - 1, 1)) * (n - 1)))
        idx = max(0, min(n - 1, idx))

        cx = int(round(trace_x[idx]))
        cy = int(round(trace_y[idx]))

        draw.rectangle(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(color[0], color[1], color[2], alpha),
            outline=(color[0], color[1], color[2], 255),
            width=line_width,
        )

        mapped_points.append(
            {
                "rank": rank,
                "img_x": cx,
                "img_y": cy,
                "score": float(score),
            }
        )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out, mapped_points