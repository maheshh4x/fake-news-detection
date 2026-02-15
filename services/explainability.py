CLICKBAIT_WORDS = [
    "shocking", "breaking", "unbelievable", "miracle", "secret", "exposed", "truth",
    "you won’t believe", "you wont believe", "must see", "viral", "insane",
    "cure", "instant", "guaranteed", "doctors hate", "trick"
]

def clickbait_score(text: str):
    text_low = text.lower()
    hits = []

    for w in CLICKBAIT_WORDS:
        if w in text_low:
            hits.append(w)

    score = min(100, len(hits) * 15)

    level = "Low"
    if score >= 60:
        level = "High"
    elif score >= 30:
        level = "Medium"

    return score, level, list(set(hits))


def explain_prediction(real_prob, fake_prob, related_count, domain_status, clickbait_level):
    reasons = []

    if fake_prob > real_prob:
        reasons.append("Model confidence leans towards FAKE based on text patterns.")
    else:
        reasons.append("Model confidence leans towards REAL based on text patterns.")

    if related_count >= 2:
        reasons.append("Multiple related sources found → supports REAL / verified information.")
    elif related_count == 0:
        reasons.append("No related sources found → could be unverified or suspicious.")

    if domain_status == "trusted":
        reasons.append("Domain is trusted → increases credibility.")
    else:
        reasons.append("Domain is unknown → credibility not confirmed.")

    if clickbait_level == "High":
        reasons.append("High clickbait language detected → common in fake/misleading news.")
    elif clickbait_level == "Medium":
        reasons.append("Some clickbait signals detected.")
    else:
        reasons.append("Low clickbait language → more natural news writing style.")

    return reasons
