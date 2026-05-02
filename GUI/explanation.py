def build_text_explanation_v2(model_name, row, threshold, retrieved_df):
    pred_label = int(row["pred_label"])
    true_label = int(row["true_label"])
    prob_abn = float(row["prob_abnormal"])

    pred_name = CLASS_NAMES[pred_label]
    true_name = CLASS_NAMES[true_label]

    same_pred_ratio = float((retrieved_df["true_label"].values == pred_label).mean()) if len(retrieved_df) > 0 else 0.0
    same_true_ratio = float((retrieved_df["true_label"].values == true_label).mean()) if len(retrieved_df) > 0 else 0.0
    abnormal_ratio = float((retrieved_df["true_label"].values == 1).mean()) if len(retrieved_df) > 0 else 0.0

    if abs(prob_abn - threshold) < 0.05:
        conf_note = "This sample is close to the operating threshold, so the decision is borderline."
    elif prob_abn >= 0.80 or prob_abn <= 0.20:
        conf_note = "The probability is far from the threshold, so the decision confidence is relatively strong."
    else:
        conf_note = "The decision confidence is moderate."

    if same_pred_ratio >= 0.8:
        retrieval_note = "Most retrieved reference cases are consistent with the predicted label."
    elif same_pred_ratio == 0.0 and same_true_ratio >= 0.8 and pred_label != true_label:
        retrieval_note = "The retrieved evidence contradicts the predicted label and instead aligns with the true label, which makes this case useful for error analysis."
    elif same_pred_ratio >= 0.5:
        retrieval_note = "The retrieved evidence is partially consistent with the predicted label."
    else:
        retrieval_note = "The retrieved evidence is mixed and does not strongly support the predicted label."

    correctness_note = "The prediction is correct." if pred_label == true_label else "The prediction is incorrect, so this case is useful for error analysis."

    txt = (
        f"Model: {model_name}\n"
        f"Source: {row['source_path']}\n"
        f"True label: {true_name}\n"
        f"Predicted label: {pred_name}\n"
        f"Probability of Abnormal: {prob_abn:.6f}\n"
        f"Decision Threshold: {threshold:.4f}\n"
        f"Retrieved abnormal-case ratio: {abnormal_ratio:.2%}\n"
        f"Retrieved support ratio for predicted label: {same_pred_ratio:.2%}\n"
        f"Retrieved support ratio for true label: {same_true_ratio:.2%}\n"
        f"{correctness_note}\n"
        f"{conf_note}\n"
        f"{retrieval_note}"
    )
    return txt