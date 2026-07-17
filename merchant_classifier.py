import re

CATEGORY_RULES = {
    "قهوة": ["starbucks", "dunkin", "barn", "dose", "coffee", "cafe", "كوفي", "قهوة"],
    "مطاعم": ["hungerstation", "jahez", "mcdonald", "burger", "restaurant", "kfc", "مطعم"],
    "تسوق": ["jarir", "amazon", "noon", "zara", "mall", "namshi", "shopping"],
    "فواتير": ["stc", "zain", "mobily", "electricity", "water", "bill", "فاتورة"],
    "اشتراكات": ["netflix", "spotify", "icloud", "shahid", "osn", "subscription"],
    "وقود": ["aramco", "petrol", "gas station", "fuel", "وقود"],
    "بقالة": ["tamimi", "panda", "carrefour", "danube", "market", "supermarket", "بقالة"],
    "صحة": ["pharmacy", "nahdi", "aldawaa", "hospital", "clinic", "صيدلية"],
    "تحويلات": ["transfer", "sadad", "mada transfer", "تحويل"],
}

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def classify_transaction(merchant: str, description: str = ""):
    text = normalize_text(f"{merchant} {description}")
    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return {
                    "category": category,
                    "confidence": 0.92,
                    "method": "merchant_keyword_classifier",
                    "matched_keyword": keyword
                }
    return {
        "category": "أخرى",
        "confidence": 0.55,
        "method": "fallback_default_category",
        "matched_keyword": None
    }
