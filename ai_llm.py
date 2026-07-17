"""
Real generative-AI layer for نبراس المالي.

Everything else in this backend (merchant_classifier.py, the rebalance
math, the static recommendation catalog) is rule-based: fast, predictable,
zero cost, but not "AI" in the generative sense. This module is the part
that actually calls a language model.

It talks to a locally running Ollama server (https://ollama.com) — free,
runs on your machine, no API key, no internet dependency once the model
is pulled. If Ollama isn't running, every function fails safely and
returns a clear status the frontend can show instead of crashing.
"""

import json
import re
import hashlib
from pathlib import Path
import requests
from ai_engine import summarize_financials, predict_budget_risk, DATA_DIR

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b-instruct"  # pull with: ollama pull qwen2.5:3b-instruct
# Arabic quality note: Qwen2.5 was trained on 29+ languages including Arabic
# and reads noticeably more natural in Arabic than Llama 3.2. The 3B variant
# trades a little quality for much faster, more reliable local inference —
# worth it for short, simple outputs like these recommendations. If you have
# 8GB+ free RAM and want richer phrasing, "qwen2.5:7b-instruct" is the upgrade.

CACHE_PATH = DATA_DIR / "ai_cache.json"


def _load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _fingerprint(payload: dict) -> str:
    """A short hash of the numbers that actually matter for the prompt.
    If none of these changed since the last call, the old answer is still
    valid — no need to bother the model again."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cached_call(cache_key: str, fingerprint_payload: dict, prompt: str, parse_fn):
    """parse_fn(raw_text) -> parsed value (list of items, or plain string).
    Must raise on bad input so we can fall back cleanly."""
    cache = _load_cache()
    entry = cache.get(cache_key)
    fp = _fingerprint(fingerprint_payload)

    # Same underlying numbers as last time → instant, no Ollama call at all.
    if entry and entry.get("fingerprint") == fp:
        return {
            "source": "ollama:" + OLLAMA_MODEL,
            "generated": True,
            "reason": None,
            "value": entry["value"],
            "from_cache": True,
        }

    result = _call_ollama(prompt)
    if result["ok"]:
        try:
            parsed = parse_fn(result["text"])
        except Exception:
            parsed = None
        if parsed:
            cache[cache_key] = {"fingerprint": fp, "value": parsed}
            _save_cache(cache)
            return {"source": "ollama:" + OLLAMA_MODEL, "generated": True, "reason": None, "value": parsed, "from_cache": False}
        result = {"ok": False, "reason": "الموديل رجع رد بصيغة غير متوقعة."}

    # Ollama failed, timed out, or returned something we couldn't parse —
    # fall back to the last good answer we have, rather than a hard error
    # in a live demo.
    if entry:
        return {
            "source": "ollama:" + OLLAMA_MODEL,
            "generated": True,
            "reason": f"(نسخة سابقة محفوظة — تعذر توليد نسخة جديدة: {result['reason']})",
            "value": entry["value"],
            "from_cache": True,
        }

    return {"source": "ollama:" + OLLAMA_MODEL, "generated": False, "reason": result["reason"], "value": None, "from_cache": False}


def _call_ollama(prompt: str, timeout: int = 120):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return {
                "ok": False,
                "reason": f"موديل '{OLLAMA_MODEL}' مو موجود عندك بعد. شغّلي بالتيرمنال: "
                          f"ollama pull {OLLAMA_MODEL} — وانتظري لين يخلص التحميل، ثم جربي الزر مرة ثانية.",
            }
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if not text:
            return {"ok": False, "reason": "الموديل رجع رد فاضي."}
        return {"ok": True, "text": text}
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "reason": "ما قدرنا نوصل لسيرفر Ollama على http://localhost:11434. "
                      "تأكد إنك مشغّل Ollama محليًا (شغّلي: ollama serve، أو افتحي تطبيق Ollama).",
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "reason": "الموديل أخذ وقت أطول من المتوقع. جربي موديل أصغر أو حاولي مرة ثانية."}
    except Exception as e:
        return {"ok": False, "reason": f"خطأ غير متوقع: {e}"}


def _parse_recommendation_items(raw_text: str):
    """The model is asked for a JSON array. Small local models sometimes wrap
    it in prose or code fences anyway, so we salvage the JSON substring
    before parsing, then fall back to splitting numbered lines if that
    still fails — either path returns [{"title", "detail"}, ...] or raises."""
    text = raw_text.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            items = json.loads(candidate)
            cleaned = [
                {"title": str(i.get("title", "")).strip(), "detail": str(i.get("detail", "")).strip()}
                for i in items if isinstance(i, dict) and i.get("title") and i.get("detail")
            ]
            if cleaned:
                return cleaned[:4]
        except Exception:
            pass

    # Fallback: split "1. ..." / "2. ..." style numbered lines, title = first
    # few words, detail = the rest.
    lines = re.split(r"\n?\s*\d+[.\-)]\s*", text)
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        words = line.split()
        title = " ".join(words[:4])
        detail = line
        items.append({"title": title, "detail": detail})
    return items[:4] if items else None


def generate_ai_recommendations():
    """Real per-user recommendations generated from actual spending data."""
    s = summarize_financials()
    r = predict_budget_risk()
    top_categories = sorted(s["by_category"].items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = "، ".join(f"{cat}: {round(amount)} ريال" for cat, amount in top_categories)

    prompt = f"""أنت مستشار مالي شخصي يتحدث بالعربية الفصحى البسيطة. بيانات المستخدم الفعلية هذا الشهر:

- الدخل الشهري: {s['profile']['monthly_income']} ريال
- الميزانية الشهرية المحددة: {s['profile']['monthly_budget']} ريال
- إجمالي الصرف حتى الآن: {round(s['total_spending'])} ريال
- أعلى فئات الصرف: {top_str}
- احتمالية تجاوز الميزانية بنهاية الشهر: {r['risk_percentage']}%
- الهدر المحتمل المكتشف: {round(s['waste_amount'])} ريال

اكتب 3 توصيات مالية قصيرة، مبنية فعليًا على الأرقام أعلاه وليست نصائح عامة.
أخرج ردك بصيغة JSON فقط، بدون أي نص قبله أو بعده، بالشكل التالي بالضبط:
[
  {{"title": "عنوان قصير للتوصية (٣-٥ كلمات)", "detail": "شرح التوصية بجملة إلى جملتين"}},
  {{"title": "...", "detail": "..."}},
  {{"title": "...", "detail": "..."}}
]"""

    fingerprint_payload = {
        "income": s["profile"]["monthly_income"],
        "budget": s["profile"]["monthly_budget"],
        "spending": round(s["total_spending"]),
        "top": top_str,
        "risk": r["risk_percentage"],
        "waste": round(s["waste_amount"]),
    }
    return _cached_call("recommendations", fingerprint_payload, prompt, _parse_recommendation_items)


def ask_finance_question(question: str):
    """Free-form Q&A grounded in the user's real numbers (a tiny RAG pattern:
    real data retrieved first, then handed to the model as context)."""
    s = summarize_financials()
    r = predict_budget_risk()
    context = f"""بيانات المستخدم الفعلية:
- الدخل الشهري: {s['profile']['monthly_income']} ريال
- الميزانية الشهرية: {s['profile']['monthly_budget']} ريال
- الصرف حتى الآن: {round(s['total_spending'])} ريال
- تفصيل الصرف حسب الفئة: {s['by_category']}
- احتمالية تجاوز الميزانية: {r['risk_percentage']}%
- الهدر المحتمل: {round(s['waste_amount'])} ريال
"""
    prompt = f"""أنت مساعد مالي شخصي. استخدم بيانات المستخدم الفعلية التالية للإجابة على سؤاله بدقة وباختصار (لا تتجاوز 4 أسطر). لا تخترع أرقام غير موجودة أدناه.

{context}

سؤال المستخدم: {question}

الإجابة:"""

    cache_key = "ask:" + hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]
    fingerprint_payload = {
        "question": question.strip(),
        "income": s["profile"]["monthly_income"],
        "spending": round(s["total_spending"]),
        "by_category": s["by_category"],
        "risk": r["risk_percentage"],
        "waste": round(s["waste_amount"]),
    }
    return _cached_call(cache_key, fingerprint_payload, prompt, lambda t: t.strip())
