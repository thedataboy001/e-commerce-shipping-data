from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import gradio as gr
import pandas as pd
import joblib
from src.feature_eng_pipeline import preprocess_data_inference 
from src.utils_and_constants import ARTIFACTS_DIR

# -----------------------
# Config: how to load model
# -----------------------

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"

# -----------------------
# Pydantic schema
# -----------------------

class ShipmentData(BaseModel):
    Warehouse_block: Literal["A", "B", "C", "D", "E"]  
    Mode_of_Shipment: Literal["Ship", "Flight", "Road"] 
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: int
    Prior_purchases: int
    Product_importance: Literal["low", "medium", "high"]  
    Gender: Literal["M", "F"]   
    Discount_offered: int
    Weight_in_gms: float

# -----------------------
# Load model once at startup
# -----------------------

def load_model():
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# -----------------------
# Shared prediction helper
# -----------------------

def predict_from_payload(payload: ShipmentData):
    # Convert request to DataFrame
    raw_df = pd.DataFrame([payload.dict()])

    # Apply same preprocessing as training
    X_processed = preprocess_data_inference(raw_df)

    # Predict
    y_pred = model.predict(X_processed)[0]
    y_proba = model.predict_proba(X_processed)[0, 1]  # probability of class 1 (late)

    # Remember: 1 = NOT delivered on time, 0 = delivered on time
    result = {
        "prediction": int(y_pred),
        "prediction_label": "Late delivery" if y_pred == 1 else "On-time delivery",
        "late_probability": float(y_proba),
        "on_time_probability": float(1 - y_proba),
    }

    return result

# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(title="E-commerce Shipping Delay API")

@app.get("/")
def root():
    return {"status": "ok", "message": "E-commerce shipping prediction API is running."}

@app.post("/predict")
def predict_api(data: ShipmentData):
    """
    FastAPI endpoint for JSON-based inference.
    """
    return predict_from_payload(data)

# -----------------------
# Gradio interface
# -----------------------

def gradio_predict(
    Warehouse_block,
    Mode_of_Shipment,
    Customer_care_calls,
    Customer_rating,
    Cost_of_the_Product,
    Prior_purchases,
    Product_importance,
    Gender,
    Discount_offered,
    Weight_in_gms,
):
    payload = ShipmentData(
        Warehouse_block=Warehouse_block,
        Mode_of_Shipment=Mode_of_Shipment,
        Customer_care_calls=int(Customer_care_calls),
        Customer_rating=int(Customer_rating),
        Cost_of_the_Product=int(Cost_of_the_Product),
        Prior_purchases=int(Prior_purchases),
        Product_importance=Product_importance,
        Gender=Gender,
        Discount_offered=int(Discount_offered),
        Weight_in_gms=float(Weight_in_gms),
    )

    result = predict_from_payload(payload)

    return (
        result["prediction_label"],
        result["late_probability"],
        result["on_time_probability"],
    )


iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(["A", "B", "C", "D", "E"], label="Warehouse block"),
        gr.Dropdown(["Ship", "Flight", "Road"], label="Mode of Shipment"),
        gr.Slider(1, 10, step=1, label="Customer care calls"),
        gr.Slider(1, 5, step=1, label="Customer rating"),
        gr.Number(label="Cost of the Product"),
        gr.Slider(0, 20, step=1, label="Prior purchases"),
        gr.Dropdown(["low", "medium", "high"], label="Product importance"),
        gr.Dropdown(["M", "F"], label="Gender"),
        gr.Number(label="Discount offered"),
        gr.Number(label="Weight in grams"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Probability: Late delivery (class 1)"),
        gr.Number(label="Probability: On-time delivery (class 0)"),
    ],

    title="E-commerce Shipping Delay Prediction",
    description="Predict whether a shipment will arrive on time or late.",
)

# Mount Gradio inside FastAPI at /gradio
app = gr.mount_gradio_app(app, iface, path="/gradio")