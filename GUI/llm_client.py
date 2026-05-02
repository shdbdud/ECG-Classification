from openai import OpenAI

def get_deepseek_client():
    api_key = "sk-458db2f575e540269dadba30967b66ac"
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

def rewrite_explanation_with_deepseek(evidence_payload: dict) -> str:
    client = get_deepseek_client()

    system_prompt = (
        "You are a careful ECG AI explanation writer. "
        "Use only the provided evidence. "
        "Do not add new medical facts, diagnoses, treatment advice, or unsupported claims. "
        "If retrieval evidence is weak or contradictory, say so clearly. "
        "At the end, provide a brief recommendation limited to review-oriented guidance, "
        "provide medication, treatment, or definitive clinical decisions."
    )

    user_prompt = f"""
Write one concise grounded explanation for this ECG AI result.

Rules:
- Use only the evidence provided below.
- Do not invent any diagnosis details, findings, or clinical history.
- State uncertainty clearly when the evidence is weak, limited, or contradictory.
- Keep the tone professional, clear, and concise.
- End with a section titled "Recommendation:".
- The Recommendation section should be about 80–100 words.
- The Recommendation section must be conservative, review-oriented, and based only on the available evidence.
- The Recommendation section may suggest careful image review, correlation with clinical assessment, or further professional evaluation when appropriate.
- Do not suggest drugs, treatment plans, emergency actions, or unsupported medical conclusions.
- Do not overstate confidence. Avoid definitive language unless the evidence is strong and consistent.
Evidence:
{evidence_payload}
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        stream=False,
    )
    return resp.choices[0].message.content.strip()