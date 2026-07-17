import csv, json
from pathlib import Path
from collections import defaultdict
import joblib
from merchant_classifier import classify_transaction
from train_model import train_and_save_model, MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_profile():
    return json.loads((DATA_DIR / "user_profile.json").read_text(encoding="utf-8"))

def load_transactions():
    rows = []
    with open(DATA_DIR / "transactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = classify_transaction(row.get("merchant", ""), row.get("description", ""))
            row["amount"] = float(row["amount"])
            row["category"] = c["category"]
            row["category_confidence"] = c["confidence"]
            row["classification_method"] = c["method"]
            row["matched_keyword"] = c["matched_keyword"]
            row["month"] = row["date"][:7]
            rows.append(row)
    return rows

def load_subscriptions():
    rows = []
    with open(DATA_DIR / "subscriptions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["monthly_cost"] = float(row["monthly_cost"])
            row["last_used_days"] = int(row["last_used_days"])
            rows.append(row)
    return rows

def summarize_financials():
    profile = load_profile()
    txs = load_transactions()
    subs = load_subscriptions()
    current_txs = [t for t in txs if t["month"] == "2026-07"]
    previous_txs = [t for t in txs if t["month"] != "2026-07"]
    by_category, previous_by_category = defaultdict(float), defaultdict(float)
    classified_count = 0

    for t in current_txs:
        by_category[t["category"]] += t["amount"]
        if t["category"] != "أخرى":
            classified_count += 1

    for t in previous_txs:
        previous_by_category[t["category"]] += t["amount"]

    total_spending = sum(t["amount"] for t in current_txs)
    current_day = profile["current_day_of_month"]
    daily_spending_rate = total_spending / current_day
    projected_month_end_spending = daily_spending_rate * 30
    expected_saving = profile["monthly_income"] - projected_month_end_spending
    saving_gap = profile["saving_goal"] - expected_saving
    budget_usage_ratio = total_spending / profile["monthly_budget"]

    waste_items, waste_amount = [], 0.0

    for cat, amount in by_category.items():
        budget = profile["budgets"].get(cat, profile["budgets"].get("أخرى", 0))
        if amount > budget:
            over = amount - budget
            waste_amount += over
            waste_items.append({"type": "category_over_budget", "category": cat, "amount": round(over, 2), "message": f"صرف {cat} تجاوز الميزانية المحددة بمقدار {round(over)} ريال."})

    months = set(t["month"] for t in previous_txs)
    month_count = max(len(months), 1)
    for cat, amount in by_category.items():
        avg_prev = previous_by_category[cat] / month_count
        if avg_prev > 0 and amount > avg_prev * 1.35:
            waste_items.append({"type": "above_historical_average", "category": cat, "amount": round(amount - avg_prev, 2), "message": f"صرف {cat} أعلى من متوسط الأشهر السابقة بنسبة تقريبية {round(((amount/avg_prev)-1)*100)}%."})

    unused_subscriptions = []
    for s in subs:
        if s["last_used_days"] >= 30:
            waste_amount += s["monthly_cost"]
            unused_subscriptions.append(s)
            waste_items.append({"type": "unused_subscription", "category": "اشتراكات", "amount": s["monthly_cost"], "message": f"اشتراك {s['name']} غير مستخدم منذ {s['last_used_days']} يوم وقد يسبب هدرًا شهريًا."})

    features = {
        "monthly_income": profile["monthly_income"],
        "saving_goal": profile["saving_goal"],
        "current_day": current_day,
        "total_spending": total_spending,
        "monthly_budget": profile["monthly_budget"],
        "budget_usage_ratio": budget_usage_ratio,
        "daily_spending_rate": daily_spending_rate,
        "projected_month_end_spending": projected_month_end_spending,
        "expected_saving": expected_saving,
        "saving_gap": saving_gap,
        "restaurants_spending": by_category.get("مطاعم", 0.0),
        "coffee_spending": by_category.get("قهوة", 0.0),
        "shopping_spending": by_category.get("تسوق", 0.0),
        "subscriptions_spending": by_category.get("اشتراكات", 0.0),
        "waste_amount": waste_amount
    }

    return {"profile": profile, "transactions": current_txs, "all_transactions": txs, "subscriptions": subs, "by_category": dict(by_category), "total_spending": total_spending, "projected_month_end_spending": projected_month_end_spending, "expected_saving": expected_saving, "saving_gap": saving_gap, "budget_usage_ratio": budget_usage_ratio, "waste_amount": waste_amount, "waste_items": waste_items, "unused_subscriptions": unused_subscriptions, "classification_rate": classified_count / max(len(current_txs), 1), "features": features}

def get_model_payload():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)

def predict_budget_risk():
    summary = summarize_financials()
    payload = get_model_payload()
    feature_vector = [[summary["features"][f] for f in payload["features"]]]
    probability = float(payload["model"].predict_proba(feature_vector)[0][1])
    level = "مرتفع" if probability >= 0.75 else ("متوسط" if probability >= 0.45 else "منخفض")
    causes = []
    if summary["budget_usage_ratio"] >= 0.75: causes.append("استهلاك نسبة عالية من الميزانية")
    if summary["saving_gap"] > 0: causes.append("الادخار المتوقع أقل من الهدف")
    if summary["waste_amount"] > 0: causes.append("وجود هدر محتمل في بعض التصنيفات أو الاشتراكات")
    return {"model_name": payload["model_name"], "target": payload["target"], "demo_accuracy": round(payload["accuracy"], 3), "risk_probability": round(probability, 3), "risk_percentage": round(probability * 100, 1), "risk_level": level, "will_exceed_budget": bool(probability >= 0.5), "top_causes": causes, "features": summary["features"]}

def build_dashboard():
    s, r = summarize_financials(), predict_budget_risk()
    return {"user": s["profile"]["name"], "monthly_income": s["profile"]["monthly_income"], "monthly_budget": s["profile"]["monthly_budget"], "saving_goal": s["profile"]["saving_goal"], "total_spending": round(s["total_spending"], 2), "expected_saving": round(s["expected_saving"], 2), "projected_month_end_spending": round(s["projected_month_end_spending"], 2), "budget_usage_percentage": round(s["budget_usage_ratio"] * 100, 1), "risk_level": r["risk_level"], "risk_percentage": r["risk_percentage"], "classification_rate": round(s["classification_rate"] * 100, 1), "waste_amount": round(s["waste_amount"], 2), "top_categories": sorted([{"category": k, "amount": round(v, 2)} for k, v in s["by_category"].items()], key=lambda x: x["amount"], reverse=True)}

def build_alerts():
    s, r = summarize_financials(), predict_budget_risk()
    alerts = []
    if r["risk_probability"] >= 0.45:
        alerts.append({"severity": r["risk_level"], "title": "مؤشر خطر مالي مبكر", "message": f"احتمالية تجاوز الميزانية أو انخفاض الادخار المتوقع وصلت إلى {r['risk_percentage']}%.", "action": "خفّض الصرف في أعلى تصنيفين خلال الأسبوع القادم وراجع هدف الادخار."})
    for item in s["waste_items"][:5]:
        alerts.append({"severity": "متوسط", "title": "هدر محتمل", "message": item["message"], "action": "راجع هذا التصنيف وحدد سقف صرف أقل لبقية الشهر."})
    return alerts or [{"severity": "منخفض", "title": "الوضع المالي مستقر", "message": "لا توجد مؤشرات خطر عالية حاليًا.", "action": "استمر على نفس نمط الصرف."}]

def build_budget_recommendation():
    s, profile = summarize_financials(), summarize_financials()["profile"]
    recommendations = {}
    for cat, budget in profile["budgets"].items():
        spent = s["by_category"].get(cat, 0)
        recommended = max(budget * 0.75, spent * 0.75) if spent > budget else (budget * 0.9 if spent < budget * 0.5 else budget)
        recommendations[cat] = {"current_budget": round(budget, 2), "spent": round(spent, 2), "recommended_budget": round(recommended, 2), "status": "تجاوز" if spent > budget else "ضمن الميزانية"}
    return {"saving_goal": profile["saving_goal"], "expected_saving": round(s["expected_saving"], 2), "recommendations": recommendations, "note": "الميزانية المقترحة تعتمد على الدخل، الهدف الادخاري، الصرف الحالي، والتصنيفات الأكثر استنزافًا."}

def simulate_purchase(amount: float, category: str = "تسوق"):
    s = summarize_financials()
    new_spending = s["total_spending"] + amount
    new_projected = (new_spending / s["profile"]["current_day_of_month"]) * 30
    new_expected_saving = s["profile"]["monthly_income"] - new_projected
    over_budget = new_projected > s["profile"]["monthly_budget"]
    saving_gap = s["profile"]["saving_goal"] - new_expected_saving
    advice = "الأفضل تأجيل العملية أو تخفيضها لأنها قد ترفع خطر تجاوز الميزانية أو تقلل الادخار المتوقع." if over_budget or saving_gap > 0 else "العملية آمنة نسبيًا."
    return {"purchase_amount": amount, "category": category, "new_projected_month_end_spending": round(new_projected, 2), "new_expected_saving": round(new_expected_saving, 2), "over_budget": over_budget, "saving_gap": round(saving_gap, 2), "advice": advice}

def apply_rebalance_plan(plan: list):
    """Persists a suggested essential/daily/annual/saving/invest split.

    `plan` is a list of {key, name, percent, amount} dicts coming from the
    frontend. We don't have a clean 1:1 mapping to the per-category budgets
    used elsewhere (مطاعم, قهوة, ...), so we store the bucketed plan as-is
    and use its "saving" bucket to update the user's saving goal, which is
    the number the rest of the app (dashboard, risk model) actually reads.
    """
    profile = load_profile()
    total_pct = sum(p.get("percent", 0) for p in plan)
    if round(total_pct) != 100:
        return {"applied": False, "reason": "الحصص لا تجمع إلى 100%", "total_percent": total_pct}

    income = profile["monthly_income"]
    saving_item = next((p for p in plan if p.get("key") == "saving"), None)
    if saving_item:
        profile["saving_goal"] = round(income * saving_item["percent"] / 100)

    profile["rebalance_plan"] = {
        "buckets": plan,
        "applied_on_day": profile.get("current_day_of_month"),
    }

    (DATA_DIR / "user_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"applied": True, "saving_goal": profile["saving_goal"], "plan": plan}


CUSTOM_ITEM_SECTIONS = ("monthly", "annual", "saving")

def get_custom_items():
    profile = load_profile()
    items = profile.get("custom_items", {})
    return {s: items.get(s, []) for s in CUSTOM_ITEM_SECTIONS}

def _save_profile(profile):
    (DATA_DIR / "user_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def add_custom_item(section: str, name: str, planned: float):
    if section not in CUSTOM_ITEM_SECTIONS:
        return {"added": False, "reason": f"القسم غير مدعوم. الأقسام المتاحة: {', '.join(CUSTOM_ITEM_SECTIONS)}"}
    name = (name or "").strip()
    if not name:
        return {"added": False, "reason": "لازم تدخلين اسم للبند"}
    profile = load_profile()
    profile.setdefault("custom_items", {s: [] for s in CUSTOM_ITEM_SECTIONS})
    profile["custom_items"].setdefault(section, [])
    existing = [i for i in profile["custom_items"][section] if i["name"] != name]
    existing.append({"name": name, "planned": round(float(planned), 2)})
    profile["custom_items"][section] = existing
    _save_profile(profile)
    return {"added": True, "section": section, "items": profile["custom_items"][section]}

def remove_custom_item(section: str, name: str):
    if section not in CUSTOM_ITEM_SECTIONS:
        return {"removed": False, "reason": "القسم غير مدعوم"}
    profile = load_profile()
    profile.setdefault("custom_items", {s: [] for s in CUSTOM_ITEM_SECTIONS})
    before = profile["custom_items"].get(section, [])
    after = [i for i in before if i["name"] != name]
    profile["custom_items"][section] = after
    _save_profile(profile)
    return {"removed": len(after) != len(before), "section": section, "items": after}


def generate_financial_report():
    d, a, r = build_dashboard(), build_alerts(), predict_budget_risk()
    return {"title": "تقرير نبراس المالي", "summary": f"حتى الآن بلغ إجمالي مصروفاتك {d['total_spending']} ريال، ومن المتوقع أن يصل الصرف بنهاية الشهر إلى {d['projected_month_end_spending']} ريال. مستوى الخطر الحالي {d['risk_level']} بنسبة {d['risk_percentage']}%.", "recommendation": "ننصح بمراجعة التصنيفات الأعلى صرفًا وتقليل المصروفات غير الأساسية خلال الأسبوع القادم، مع إعطاء أولوية للحفاظ على هدف الادخار الشهري.", "top_causes": r["top_causes"], "main_alert": a[0] if a else None}
