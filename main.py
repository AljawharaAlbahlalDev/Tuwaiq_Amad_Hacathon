from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ai_engine import build_dashboard, build_alerts, build_budget_recommendation, load_transactions, predict_budget_risk, simulate_purchase, generate_financial_report, get_model_payload, summarize_financials, apply_rebalance_plan, get_custom_items, add_custom_item, remove_custom_item
from ai_llm import generate_ai_recommendations, ask_finance_question
from merchant_classifier import classify_transaction

app = FastAPI(title="نبراس المالي API", description="مساعد مالي ذكي لتحليل السلوك المالي والتنبؤ بخطر تجاوز الميزانية", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PurchaseRequest(BaseModel):
    amount: float
    category: str = "تسوق"

class ClassifyRequest(BaseModel):
    merchant: str
    description: str = ""

class RebalanceBucket(BaseModel):
    key: str
    name: str
    percent: float
    note: str = ""

class RebalanceRequest(BaseModel):
    plan: List[RebalanceBucket]

class CustomItemRequest(BaseModel):
    section: str
    name: str
    planned: float

class CustomItemDeleteRequest(BaseModel):
    section: str
    name: str

class AskQuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"app": "نبراس المالي", "status": "running", "message": "Backend is running. Open /docs to test APIs."}

@app.get("/api/dashboard")
def dashboard(): return build_dashboard()

@app.get("/api/alerts")
def alerts(): return build_alerts()

@app.get("/api/budget")
def budget(): return build_budget_recommendation()

@app.get("/api/transactions")
def transactions(): return load_transactions()

@app.get("/api/predict-risk")
def predict_risk(): return predict_budget_risk()

@app.get("/api/model-info")
def model_info():
    p = get_model_payload()
    return {"model_name": p["model_name"], "target": p["target"], "features": p["features"], "demo_accuracy": round(p["accuracy"], 3), "note": "النموذج مدرب على بيانات اصطناعية لأغراض النموذج الأولي."}

@app.post("/api/classify-merchant")
def classify_merchant(req: ClassifyRequest): return classify_transaction(req.merchant, req.description)

@app.get("/api/waste")
def waste():
    s = summarize_financials()
    return {"waste_amount": round(s["waste_amount"], 2), "waste_items": s["waste_items"], "unused_subscriptions": s["unused_subscriptions"]}

@app.post("/api/simulate-purchase")
def simulate(req: PurchaseRequest): return simulate_purchase(req.amount, req.category)

@app.get("/api/report")
def report(): return generate_financial_report()

@app.post("/api/budget/rebalance")
def rebalance(req: RebalanceRequest):
    return apply_rebalance_plan([b.dict() for b in req.plan])

@app.get("/api/budget/custom-items")
def custom_items(): return get_custom_items()

@app.post("/api/budget/custom-item")
def create_custom_item(req: CustomItemRequest):
    return add_custom_item(req.section, req.name, req.planned)

@app.post("/api/budget/custom-item/delete")
def delete_custom_item(req: CustomItemDeleteRequest):
    return remove_custom_item(req.section, req.name)

@app.get("/api/ai/recommendations")
def ai_recommendations():
    return generate_ai_recommendations()

@app.post("/api/ai/ask")
def ai_ask(req: AskQuestionRequest):
    return ask_finance_question(req.question)
