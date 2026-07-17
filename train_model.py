import random
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path(__file__).parent / "budget_risk_model.joblib"

FEATURES = [
    "monthly_income", "saving_goal", "current_day", "total_spending",
    "monthly_budget", "budget_usage_ratio", "daily_spending_rate",
    "projected_month_end_spending", "expected_saving", "saving_gap",
    "restaurants_spending", "coffee_spending", "shopping_spending",
    "subscriptions_spending", "waste_amount"
]

def generate_training_data(n=1200, seed=42):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        monthly_income = random.choice([7000, 9000, 12000, 15000, 18000, 22000])
        saving_goal = random.randint(500, 4000)
        current_day = random.randint(5, 28)
        monthly_budget = max(monthly_income - saving_goal - random.randint(500, 1500), 2500)
        normal_spending = monthly_budget * (current_day / 30)
        total_spending = normal_spending * random.uniform(0.65, 1.65)
        restaurants = total_spending * random.uniform(0.08, 0.22)
        coffee = total_spending * random.uniform(0.02, 0.08)
        shopping = total_spending * random.uniform(0.08, 0.28)
        subscriptions = random.uniform(50, 350)
        budget_usage_ratio = total_spending / monthly_budget
        daily_spending_rate = total_spending / current_day
        projected_month_end_spending = daily_spending_rate * 30
        expected_saving = monthly_income - projected_month_end_spending
        saving_gap = saving_goal - expected_saving
        waste_amount = max(0, restaurants - 900) + max(0, coffee - 350) + max(0, shopping - 1200) + max(0, subscriptions - 250)
        risk = int(projected_month_end_spending > monthly_budget or expected_saving < saving_goal or waste_amount > 800)
        X.append([monthly_income, saving_goal, current_day, total_spending, monthly_budget, budget_usage_ratio, daily_spending_rate, projected_month_end_spending, expected_saving, saving_gap, restaurants, coffee, shopping, subscriptions, waste_amount])
        y.append(risk)
    return np.array(X), np.array(y)

def train_and_save_model():
    X, y = generate_training_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    model = RandomForestClassifier(n_estimators=120, random_state=7, max_depth=7)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    payload = {"model": model, "features": FEATURES, "accuracy": float(acc), "model_name": "RandomForestClassifier", "target": "budget_overrun_or_saving_gap_risk"}
    joblib.dump(payload, MODEL_PATH)
    return payload

if __name__ == "__main__":
    payload = train_and_save_model()
    print(f"Model saved to {MODEL_PATH}")
    print(f"Training demo accuracy: {payload['accuracy']:.2f}")
