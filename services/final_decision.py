def final_verdict(model_result, confidence, related_articles_count, domain_status="unknown"):

    # High confidence thresholds
    REAL_HIGH = 0.75
    FAKE_HIGH = 0.75

    # ✅ Strong REAL
    if model_result == "Real News" and confidence >= REAL_HIGH:
        return "REAL"

    # ❌ Strong FAKE
    if model_result == "Fake News" and confidence >= FAKE_HIGH:
        return "FAKE"

    # 🔍 If related articles exist → likely real
    if related_articles_count >= 2:
        return "REAL"

    # 🚨 Suspicious domain + low verification
    if domain_status == "suspicious" and related_articles_count == 0:
        return "FAKE"

    # ⚠️ Otherwise uncertain
    return "UNCERTAIN"
