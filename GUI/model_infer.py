import numpy as np
import cv2
import pywt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

from app_config import DEVICE, CLASS_NAMES, CKPTS, THR_MAP, MODEL_RUNTIME_CONFIG
from model_defs import DSRFromImages, EnhancedDSResSE, PureCNN, CNNLSTM, PureLSTM,OriginalDSRAblation

IMG_SIZE = 224
USE_CWT = True  

MODEL_ZOO = {
    "Original_DSR": DSRFromImages,
    "Enhanced_DSR_SE": EnhancedDSResSE,
    "Pure_CNN": PureCNN,
    "CNN_LSTM": CNNLSTM,
    "Pure_LSTM": PureLSTM,
    "Original_DSR_Attn_Proto": DSRFromImages,
}


def default_infer_transform():
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def generate_cwt_scalogram(signal_1d, scales=None, wavelet='morl', size=(224, 224)):
    if scales is None:
        scales = np.arange(4, 65)
    coeffs, _ = pywt.cwt(signal_1d, scales, wavelet)
    scalogram = np.log1p(np.abs(coeffs))
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)
    scalogram = cv2.resize(scalogram, size, interpolation=cv2.INTER_LINEAR)
    return scalogram.astype(np.float32)


def crop_rhythm_strip(img_pil: Image.Image, bottom_ratio: float = 0.32) -> Image.Image:
    arr = np.array(img_pil)
    h, w = arr.shape[:2]
    y0 = int(h * (1 - bottom_ratio))
    return Image.fromarray(arr[y0:h, :])


def suppress_grid_gray(gray_u8: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    inv = 255 - g
    bw = cv2.adaptiveThreshold(inv, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               31, -5)
    h, w = bw.shape
    kx = max(10, w // 80)
    ky = max(10, h // 80)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    grid = cv2.bitwise_or(horiz, vert)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    out = gray_u8.copy()
    out[grid > 0] = np.clip(out[grid > 0] + 60, 0, 255)
    return out


def extract_ecg_from_image(image_pil, band_ratio: float = 0.55, return_meta: bool = False):
    img0 = image_pil.convert("L")
    arr0 = np.array(img0, dtype=np.uint8)
    orig_h, orig_w = arr0.shape[:2]

    # 1) 裁底部长 rhythm strip
    rhythm_y0 = int(orig_h * (1 - 0.32))
    rhythm_y1 = orig_h
    rhythm = arr0[rhythm_y0:rhythm_y1, :]

    gray = suppress_grid_gray(rhythm)

    # 2) 左右裁剪
    h_r, w_r = gray.shape
    local_x0 = int(w_r * 0.06)
    local_x1 = int(w_r * 0.94)
    gray = gray[:, local_x0:local_x1]

    crop_x0 = local_x0
    crop_x1 = local_x1

    h, w = gray.shape

    # 3) 中间 band 裁剪
    y0 = int(h * (0.5 - band_ratio / 2))
    y1 = int(h * (0.5 + band_ratio / 2))
    y0 = max(0, y0)
    y1 = min(h, y1)
    strip = gray[y0:y1, :]
    bh = strip.shape[0]

    inv = 255.0 - strip.astype(np.float32)
    thr = np.percentile(inv, 96)
    wgt = np.clip(inv - thr, 0, None)

    ys = np.arange(bh, dtype=np.float32)[:, None]
    denom = wgt.sum(axis=0)

    eps = 1e-6
    denom_thr = np.percentile(denom, 50)
    valid = denom > denom_thr
    denom_safe = denom + eps

    yhat = (ys * wgt).sum(axis=0) / denom_safe
    yhat[~valid] = np.nan

    x = np.arange(w, dtype=np.float32)
    good = ~np.isnan(yhat)
    if good.sum() >= 2:
        yhat = np.interp(x, x[good], yhat[good])
    else:
        yhat = np.full_like(x, bh / 2.0)

    sig = (bh / 2.0 - yhat)

    k = 31
    sig = np.convolve(sig, np.ones(k, dtype=np.float32) / k, mode="same")

    k2 = 201
    baseline = np.convolve(sig, np.ones(k2, dtype=np.float32) / k2, mode="same")
    sig = sig - baseline

    lo, hi = np.percentile(sig, [1, 99])
    sig = np.clip(sig, lo, hi)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)

        # 4) 两端裁掉再重采样
    ww = sig.shape[0]
    l = int(ww * 0.02)
    r = int(ww * 0.98)

    sig_mid = sig[l:r]
    yhat_mid = yhat[l:r]

    trace_x_full = np.linspace(crop_x0, crop_x1 - 1, ww).astype(np.float32)
    trace_x_mid = trace_x_full[l:r]

    x_old = np.linspace(0, 1, sig_mid.shape[0])
    x_new = np.linspace(0, 1, ww)

    sig = np.interp(x_new, x_old, sig_mid).astype(np.float32)
    trace_x = np.interp(x_new, x_old, trace_x_mid).astype(np.float32)
    trace_y_local = np.interp(x_new, x_old, yhat_mid).astype(np.float32)
    trace_y = rhythm_y0 + y0 + trace_y_local

    vis_meta = {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "rhythm_y0": int(rhythm_y0),
        "rhythm_y1": int(rhythm_y1),
        "crop_x0": int(crop_x0),
        "crop_x1": int(crop_x1),
        "band_y0": int(rhythm_y0 + y0),
        "band_y1": int(rhythm_y0 + y1),
        "trace_x": trace_x,
        "trace_y": trace_y,
    }

    if return_meta:
        return sig, vis_meta
    return sig


def preprocess_image(image_or_path, model_name: str = None, return_meta: bool = False):
    if isinstance(image_or_path, (str, Path)):
        img_pil = Image.open(image_or_path).convert("L")
    elif isinstance(image_or_path, Image.Image):
        img_pil = image_or_path.convert("L")
    else:
        raise TypeError("image_or_path must be a file path or PIL.Image")

    if USE_CWT:
        signal_1d, vis_meta = extract_ecg_from_image(img_pil, return_meta=True)
        scalogram = generate_cwt_scalogram(signal_1d, size=(IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(scalogram).unsqueeze(0).unsqueeze(0).float()

        cfg = MODEL_RUNTIME_CONFIG.get(model_name, {}) if model_name is not None else {}
        if cfg.get("use_cwt_norm", False):
            img_tensor = (img_tensor - 0.5) / 0.5

    else:
        tfm = default_infer_transform()
        img_tensor = tfm(img_pil).unsqueeze(0)
        vis_meta = None

    rr_tensor = torch.zeros((1, 0), dtype=torch.float32)

    if return_meta:
        return img_tensor, rr_tensor, vis_meta
    return img_tensor, rr_tensor


def load_state(ckpt_path, map_location=None):
    ckpt_path = str(ckpt_path)
    try:
        obj = torch.load(ckpt_path, map_location=map_location or DEVICE, weights_only=True)
    except Exception:
        obj = torch.load(ckpt_path, map_location=map_location or DEVICE, weights_only=False)

    if isinstance(obj, dict):
        if 'model_state_dict' in obj:
            return obj['model_state_dict']
        if 'state_dict' in obj:
            return obj['state_dict']
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError(f'Cannot parse checkpoint: {ckpt_path}')


def infer_use_rag_from_state(state_dict):
    keys = list(state_dict.keys())
    return any('rag' in k.lower() for k in keys)


def infer_rr_dim_from_state(state_dict, default_rr_dim=0):
    candidate_keys = ['rrnet.0.weight', 'rrnet.weight', 'rr_branch.0.weight', 'rr_branch.weight']
    for k in candidate_keys:
        if k in state_dict and hasattr(state_dict[k], 'shape'):
            w = state_dict[k]
            if len(w.shape) == 2:
                return int(w.shape[1])
    return default_rr_dim


def load_trained_model(model_name: str, device=DEVICE):
    state = load_state(CKPTS[model_name], map_location=device)
    use_rag = infer_use_rag_from_state(state)
    rr_dim = infer_rr_dim_from_state(state, default_rr_dim=0)

    cfg = MODEL_RUNTIME_CONFIG.get(model_name, {})
    loader = cfg.get("model_loader", "")

    if loader == "original_dsr_ablation_attn":
        model = OriginalDSRAblation(
            num_classes=2,
            rr_dim=rr_dim,
            use_attn=True,
            use_rag=use_rag,
        )
    else:
        model = MODEL_ZOO[model_name](
            num_classes=2,
            rr_dim=rr_dim,
            use_rag=use_rag,
        )

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[load_trained_model] model={model_name}")
    print(f"[load_trained_model] class={type(model).__name__}")
    print(f"[load_trained_model] loader={loader}")
    print(f"[load_trained_model] use_rag={use_rag}, rr_dim={rr_dim}")
    print(f"[load_trained_model] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    if missing:
        print("missing sample:", missing[:20])
    if unexpected:
        print("unexpected sample:", unexpected[:20])

    model.to(device)
    model.eval()
    return model

def maybe_apply_internal_rag(model, z):
    if getattr(model, 'use_rag', False) and hasattr(model, 'rag'):
        return model.rag(z)
    return z


@torch.no_grad()
def extract_embedding(model, img, rr):
    model.eval()
    # Original_DSR / DSRFromImages
    if hasattr(model, "morph") and hasattr(model, "fuse"):
        fm = model.morph(img)
        fr = model.rrnet(rr)

        # 新版：优先走 self-attention 融合
        if hasattr(model, "self_attn_fusion"):
            z = model.self_attn_fusion(fm, fr)
        else:
            # 兼容旧版
            z = torch.cat([fm, fr], dim=1)

        z = model.fuse(z)
        z = maybe_apply_internal_rag(model, z)
        return F.normalize(z.float(), p=2, dim=1)

    if hasattr(model, 'stem') and hasattr(model, 'b1_conv') and hasattr(model, 'b2_conv') and hasattr(model, 'b3_conv'):
        x = model.stem(img)
        x = model.b1_conv(x)
        x = model.b2_conv(x)
        x = model.b3_conv(x)
        x = x.view(x.size(0), -1)
        fr = model.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = model.feature_extractor(z)
        z = maybe_apply_internal_rag(model, z)
        return F.normalize(z.float(), p=2, dim=1)

    if hasattr(model, 'stem') and hasattr(model, 'conv1') and hasattr(model, 'conv2') and hasattr(model, 'conv3'):
        x = model.stem(img)
        x = model.conv1(x)
        x = model.conv2(x)
        x = model.conv3(x)
        x = x.view(x.size(0), -1)
        fr = model.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = model.feature_extractor(z)
        z = maybe_apply_internal_rag(model, z)
        return F.normalize(z.float(), p=2, dim=1)

    if hasattr(model, 'cnn') and hasattr(model, 'lstm1') and hasattr(model, 'lstm2'):
        x = model.cnn(img)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        x, _ = model.lstm1(x)
        x = model.dropout(x)
        x, _ = model.lstm2(x)
        x = x[:, -1, :]
        fr = model.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = model.feature_extractor(z)
        z = maybe_apply_internal_rag(model, z)
        return F.normalize(z.float(), p=2, dim=1)

    if hasattr(model, 'lstm1') and hasattr(model, 'lstm2') and not hasattr(model, 'cnn'):
        x = img.squeeze(1)
        x, _ = model.lstm1(x)
        x = model.dropout(x)
        x, _ = model.lstm2(x)
        x = x[:, -1, :]
        fr = model.rrnet(rr)
        z = torch.cat([x, fr], dim=1)
        z = model.feature_extractor(z)
        z = maybe_apply_internal_rag(model, z)
        return F.normalize(z.float(), p=2, dim=1)

    raise ValueError(f'Unsupported model structure for embedding extraction: {type(model).__name__}')

@torch.no_grad()
def predict_probs_and_logits(model, img, rr):
    model.eval()
    logits = model(img, rr)
    probs = F.softmax(logits.float(), dim=1)

    return {
        "logits": logits,
        "probs": probs,
        "prob_normal": probs[:, 0],
        "prob_abnormal": probs[:, 1],
        "pred_argmax": torch.argmax(probs, dim=1),
    }

@torch.no_grad()
def predict_prob_abnormal(model, img, rr):
    model.eval()
    logits = model(img, rr)
    probs = F.softmax(logits.float(), dim=1)
    return probs[:, 1]
@torch.no_grad()
def predict_model_output(model, img, rr, model_name: str):
    model.eval()
    logits = model(img, rr)
    probs = F.softmax(logits.float(), dim=1)

    prob_normal = float(probs[:, 0][0].item())
    prob_abnormal = float(probs[:, 1][0].item())

    cfg = MODEL_RUNTIME_CONFIG.get(model_name, {})
    prediction_mode = cfg.get("prediction_mode", "threshold")
    threshold = float(THR_MAP.get(model_name, 0.5))

    if prediction_mode == "argmax":
        pred_label = int(torch.argmax(probs, dim=1)[0].item())
    else:
        pred_label = int(prob_abnormal >= threshold)

    return {
        "logits": logits,
        "probs": probs,
        "prob_normal": prob_normal,
        "prob_abnormal": prob_abnormal,
        "pred_label": pred_label,
        "threshold": threshold,
        "prediction_mode": prediction_mode,
    }


@torch.no_grad()
def infer_single_image(model_name, image_or_path):
    model = load_trained_model(model_name, device=DEVICE)
    img, rr = preprocess_image(image_or_path, model_name=model_name)
    img = img.to(DEVICE)
    rr = rr.to(DEVICE)

    out = predict_model_output(model, img, rr, model_name)

    prob_normal = out["prob_normal"]
    prob_abn = out["prob_abnormal"]
    pred_label = out["pred_label"]
    threshold = out["threshold"]
    prediction_mode = out["prediction_mode"]

    pred_name = CLASS_NAMES[pred_label]
    embedding = extract_embedding(model, img, rr)[0].detach().cpu().numpy()

    if prediction_mode == "argmax":
        is_borderline = abs(prob_abn - prob_normal) < 0.10
    else:
        is_borderline = abs(prob_abn - threshold) < 0.05

    return {
        "model_name": model_name,
        "pred_label": pred_label,
        "pred_name": pred_name,
        "prob_normal": prob_normal,
        "prob_abnormal": prob_abn,
        "threshold": threshold,
        "prediction_mode": prediction_mode,
        "is_borderline": is_borderline,
        "embedding": embedding,
    }