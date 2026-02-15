def compute_credibility_score(real_prob, fake_prob, related_count, domain_status, clickbait_score):
    """
    Returns credibility score 0–100
    Higher = more trustworthy
    """

    # Base score from model prediction (Real probability)
    model_score = real_prob * 100

    # Related sources effect
    if related_count >= 3:
        verification_bonus = 25
    elif related_count == 2:
        verification_bonus = 18
    elif related_count == 1:
        verification_bonus = 10
    else:
        verification_bonus = -10

    # Trusted domain bonus
    domain_bonus = 10 if domain_status == "trusted" else 0

    # Clickbait penalty (max 30)
    clickbait_penalty = clickbait_score * 0.3

    score = model_score + verification_bonus + domain_bonus - clickbait_penalty

    # Clamp score between 0 and 100
    score = max(0, min(100, score))

    return round(score, 2)
