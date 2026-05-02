import streamlit as st
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from cam_utils import (
    GradCAM,
    get_target_layer,
    overlay_cam_on_image,
    cam_to_time_segments,
    cam_profile_to_peaks,
    draw_exact_trace_peak_circles_on_original_ecg
)

from app_config import DEVICE, CLASS_NAMES, MODEL_NAME_LIST, DATA_ROOT, THR_MAP
from model_infer import (
    load_trained_model,
    preprocess_image,
    predict_model_output,
    extract_embedding,
)
from rag_pipeline import load_rag_assets, retrieve_topk_unique
from llm_client import rewrite_explanation_with_deepseek


st.set_page_config(
    page_title="ECG RAG Explainability Demo",
    page_icon="🫀",
    layout="wide",
)

st.title("ECG Binary Classification with RAG Explainability")
st.caption("Upload an ECG image, run inference, retrieve similar cases, and view grounded explanations.")


@st.cache_resource
def get_model(model_name: str):
    return load_trained_model(model_name, device=DEVICE)


@st.cache_resource
def get_rag_assets(model_name: str):
    return load_rag_assets(model_name)


def build_gui_explanation(model_name, pred_label, prob_abn, threshold, retrieved_df):
    pred_name = CLASS_NAMES[int(pred_label)]

    if len(retrieved_df) > 0:
        same_pred_ratio = float((retrieved_df["true_label"].values == int(pred_label)).mean())
        abnormal_ratio = float((retrieved_df["true_label"].values == 1).mean())
    else:
        same_pred_ratio = 0.0
        abnormal_ratio = 0.0

    if abs(prob_abn - threshold) < 0.05:
        confidence_note = "This sample is close to the operating threshold, so the decision is borderline."
    elif prob_abn >= 0.80 or prob_abn <= 0.20:
        confidence_note = "The probability is far from the threshold, so the decision confidence is relatively strong."
    else:
        confidence_note = "The decision confidence is moderate."

    if same_pred_ratio >= 0.8:
        retrieval_note = "Most retrieved reference cases are consistent with the predicted label."
    elif same_pred_ratio >= 0.5:
        retrieval_note = "The retrieved evidence is partially consistent with the predicted label."
    else:
        retrieval_note = "The retrieved evidence does not strongly support the predicted label, so this result should be reviewed carefully."

    text = (
        f"Model: {model_name}\n"
        f"Predicted label: {pred_name}\n"
        f"Probability of Abnormal: {prob_abn:.6f}\n"
        f"Decision Threshold: {threshold:.4f}\n"
        f"Retrieved abnormal-case ratio: {abnormal_ratio:.2%}\n"
        f"Retrieved support ratio for predicted label: {same_pred_ratio:.2%}\n"
        f"{confidence_note}\n"
        f"{retrieval_note}"
    )
    return text


def build_evidence_payload(model_name, pred_label, prob_abn, threshold, retrieved_df):
    pred_name = CLASS_NAMES[int(pred_label)]

    if len(retrieved_df) > 0:
        same_pred_ratio = float((retrieved_df["true_label"].values == int(pred_label)).mean())
        abnormal_ratio = float((retrieved_df["true_label"].values == 1).mean())
    else:
        same_pred_ratio = 0.0
        abnormal_ratio = 0.0

    cols = [c for c in ["rank", "source_path", "true_label_name", "pred_label_name", "prob_abnormal", "similarity"] if c in retrieved_df.columns]
    retrieved_cases = retrieved_df[cols].copy()

    if "prob_abnormal" in retrieved_cases.columns:
        retrieved_cases["prob_abnormal"] = retrieved_cases["prob_abnormal"].map(lambda x: round(float(x), 4))
    if "similarity" in retrieved_cases.columns:
        retrieved_cases["similarity"] = retrieved_cases["similarity"].map(lambda x: round(float(x), 4))

    payload = {
        "model_name": model_name,
        "predicted_label": pred_name,
        "probability_of_abnormal": round(float(prob_abn), 6),
        "decision_threshold": round(float(threshold), 4),
        "is_borderline": abs(prob_abn - threshold) < 0.05,
        "retrieved_abnormal_case_ratio": round(abnormal_ratio, 4),
        "retrieved_support_ratio_for_predicted_label": round(same_pred_ratio, 4),
        "retrieved_cases": retrieved_cases.to_dict(orient="records"),
    }
    return payload


def resolve_case_path(source_path: str):
    p = Path(source_path)
    if p.exists():
        return p
    p2 = DATA_ROOT / source_path
    if p2.exists():
        return p2
    return None


def render_retrieved_thumbnails(retrieved_df, max_show=3):
    st.subheader("Retrieved Case Thumbnails")
    n_show = min(max_show, len(retrieved_df))
    if n_show == 0:
        st.info("No retrieved case to display.")
        return

    cols = st.columns(n_show)
    for i in range(n_show):
        row = retrieved_df.iloc[i]
        img_path = resolve_case_path(str(row["source_path"]))
        with cols[i]:
            st.caption(
                f"Rank {int(row['rank'])} | "
                f"{row.get('true_label_name', 'NA')} | "
                f"sim={float(row.get('similarity', 0.0)):.4f}"
            )
            if img_path is not None and img_path.exists():
                try:
                    img = Image.open(img_path).convert("L")
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.warning(f"Image open failed: {e}")
                    st.code(str(row["source_path"]))
            else:
                st.warning("Image path not found")
                st.code(str(row["source_path"]))


with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Choose model",
        MODEL_NAME_LIST,
        index=0 if "Enhanced_DSR_SE" not in MODEL_NAME_LIST else MODEL_NAME_LIST.index("Enhanced_DSR_SE"),
    )
    top_k = st.slider("Top-k retrieved cases", min_value=3, max_value=10, value=5, step=1)
    explanation_mode = st.selectbox(
        "Explanation mode",
        ["Template only", "DeepSeek rewrite", "Both"],
        index=0,
    )

uploaded_file = st.file_uploader(
    "Upload an ECG image",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Uploaded ECG")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        run_btn = st.button("Run inference", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Running model and retrieval..."):
                model = get_model(model_name)
                rag_assets = get_rag_assets(model_name)

                img_tensor, rr_tensor, vis_meta = preprocess_image(
                    image,
                    model_name=model_name,
                    return_meta=True,
                )
                img_tensor = img_tensor.to(DEVICE)
                rr_tensor = rr_tensor.to(DEVICE)

                with torch.no_grad():
                    out = predict_model_output(model, img_tensor, rr_tensor, model_name)

                    prob_normal = out["prob_normal"]
                    prob_abn = out["prob_abnormal"]
                    pred_label = out["pred_label"]
                    threshold = out["threshold"]
                    prediction_mode = out["prediction_mode"]

                    embedding = extract_embedding(model, img_tensor, rr_tensor)[0].detach().cpu().numpy()                                               
                # ===== Grad-CAM localization =====
                cam_map = None
                cam_overlay = None
                ecg_highlight_img = None
                mapped_segments = []
                mapped_points = []
                cam_profile = None

                try:
                    if model_name != "Pure_LSTM":
                        target_layer = get_target_layer(model_name, model)
                        grad_cam = GradCAM(model, target_layer)

                        cam_map = grad_cam.generate(
                            img_tensor=img_tensor,
                            rr_tensor=rr_tensor,
                            class_idx=pred_label
                        )

                        # 这个保留。右侧还可以展示模型在输入域上的注意力热图
                        _, cam_overlay, _ = overlay_cam_on_image(img_tensor, cam_map, alpha=0.35)

                        # 新逻辑：把 2D CAM 压成 1D 时间区段
                        cam_profile = cam_map.mean(axis=0)

                        peaks = cam_profile_to_peaks(
                            cam_profile,
                            top_n=4,
                            min_distance_ratio=0.08,
                            min_score=0.35,
                        )

                        ecg_highlight_img, mapped_points = draw_exact_trace_peak_circles_on_original_ecg(
                            image_pil=image,
                            peaks=peaks,
                            vis_meta=vis_meta,
                            cam_width=cam_map.shape[1],
                            radius=100,
                            color=(220, 38, 38),
                            alpha=55,
                            line_width=4,
                        )
                        grad_cam.remove_hooks()
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed: {e}")
                retrieved_df = retrieve_topk_unique(
                    query_embedding=embedding,
                    meta_df=rag_assets["meta_df"],
                    index_obj=rag_assets["index_obj"],
                    index_type=rag_assets["index_type"],
                    k=top_k,
                    overfetch=max(20, top_k),
                )

                template_explanation = build_gui_explanation(
                    model_name=model_name,
                    pred_label=pred_label,
                    prob_abn=prob_abn,
                    threshold=threshold,
                    retrieved_df=retrieved_df,
                )

                llm_explanation = None

                if explanation_mode in ["DeepSeek rewrite", "Both"]:
                    evidence_payload = build_evidence_payload(
                        model_name=model_name,
                        pred_label=pred_label,
                        prob_abn=prob_abn,
                        threshold=threshold,
                        retrieved_df=retrieved_df,
                    )
                    try:
                        with st.spinner("Calling DeepSeek..."):
                            llm_explanation = rewrite_explanation_with_deepseek(evidence_payload)
                        st.success("DeepSeek API called successfully.")
                    except Exception as e:
                        st.warning(f"DeepSeek rewrite failed. Fallback to template explanation. Details: {e}")
                        llm_explanation = None

            st.subheader("Prediction")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Label", CLASS_NAMES[pred_label])
            with m2:
                st.metric("P(Abnormal)", f"{prob_abn:.4f}")
            with m3:
                st.metric("Threshold", f"{threshold:.4f}")
            with m4:
                if prediction_mode == "argmax":
                    borderline_flag = abs(prob_abn - prob_normal) < 0.10
                else:
                    borderline_flag = abs(prob_abn - threshold) < 0.05

                st.metric("Borderline", "Yes" if borderline_flag else "No")
            
            
            st.subheader("Visual Localization")

            if (ecg_highlight_img is not None) or (cam_overlay is not None):
                c_cam1, c_cam2 = st.columns([1.2, 1.0])

                with c_cam1:
                    if ecg_highlight_img is not None:
                        st.caption("Suspected abnormal ECG segments on the original ECG")
                        st.image(ecg_highlight_img, use_container_width=True)
                    else:
                        st.info("Original-ECG segment visualization is not available.")

                with c_cam2:
                    if cam_overlay is not None:
                        st.caption("Grad-CAM on model input")
                        st.image(cam_overlay, use_container_width=True)

                    st.caption("Highlighted segments")
                    if mapped_segments:
                        for seg in mapped_segments:
                            st.write(
                                f"Segment {seg['rank']}: "
                                f"x={seg['img_x1']} to {seg['img_x2']}, "
                                f"score={seg['score']:.4f}"
                            )
                    else:
                        st.write("")
            else:
                st.info("Visual localization is not available for this model/output.")
    if run_btn:
        st.subheader("Retrieved Similar Cases")
        show_df = retrieved_df.copy()

        preferred_cols = [
            "rank",
            "source_path",
            "true_label_name",
            "pred_label_name",
            "prob_abnormal",
            "similarity",
        ]
        show_df = show_df[[c for c in preferred_cols if c in show_df.columns]]

        if "prob_abnormal" in show_df.columns:
            show_df["prob_abnormal"] = show_df["prob_abnormal"].map(lambda x: round(float(x), 4))
        if "similarity" in show_df.columns:
            show_df["similarity"] = show_df["similarity"].map(lambda x: round(float(x), 4))

        st.dataframe(show_df, use_container_width=True)
        render_retrieved_thumbnails(retrieved_df, max_show=min(3, top_k))

        st.subheader("Explanation")

        if explanation_mode == "Template only":
            st.text_area("Template explanation", template_explanation, height=220)

        elif explanation_mode == "DeepSeek rewrite":
            final_explanation = llm_explanation if llm_explanation else template_explanation
            st.text_area("DeepSeek explanation", final_explanation, height=220)

        else:
            c1, c2 = st.columns(2)
            with c1:
                st.text_area("Template explanation", template_explanation, height=260)
            with c2:
                final_explanation = llm_explanation if llm_explanation else template_explanation
                st.text_area("DeepSeek explanation", final_explanation, height=260)

        st.info(
            "This explanation is grounded in model output and retrieved reference cases. "
            "It is intended as decision support rather than a final diagnosis."
        )
else:
    st.info("Upload one ECG image to start.")