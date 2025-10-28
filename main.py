from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="SHAP-Guard Bank API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Load ML model and explainer
try:
    with open('models/fraud_detector.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Models not loaded - {e}")
    model = None
    explainer = None


# Pydantic models
class TransactionInput(BaseModel):
    customer_id: str
    amount: float
    transaction_type: str
    hour: int
    day: int
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    is_night: int = 0
    is_weekend: int = 0
    is_late_night: int = 0
    new_device: int = 0
    device_change: int = 0
    first_time_payee: int = 0
    voice_liveness_score: float = 0.8
    doc_anomaly_score: float = 0.1
    face_similarity: float = 0.9


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    analyst_note: str
    top_risk_factors: List[Dict]
    timestamp: str


class AlertResponse(BaseModel):
    id: str
    transaction_id: str
    fraud_probability: float
    status: str
    created_at: str


# Helper functions
def calculate_risk_level(fraud_prob: float) -> str:
    """Determine risk level based on fraud probability"""
    if fraud_prob >= 0.7:
        return "HIGH"
    elif fraud_prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def generate_evidence_card(features: Dict, model, explainer) -> Dict:
    """Generate SHAP-based evidence card"""
    # Convert features to array
    feature_array = np.array([features.get(name, 0) for name in feature_names])
    
    # Predict
    fraud_prob = model.predict_proba([feature_array])[0, 1]
    is_fraud = fraud_prob >= 0.5
    
    # Get SHAP values
    shap_vals = explainer.shap_values([feature_array])[0]
    
    # Create contributions
    contributions = [
        (name, features.get(name, 0), float(shap_vals[i])) 
        for i, name in enumerate(feature_names)
    ]
    
    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    top_drivers = contributions[:5]
    
    # Generate analyst note
    risk_factors = []
    for feature, value, shap_val in top_drivers[:3]:
        if shap_val > 0:
            risk_factors.append(f"{feature}")
    
    analyst_note = f"{'🚨 HIGH RISK' if is_fraud else '✅ LOW RISK'} "
    if risk_factors:
        analyst_note += f"due to: {', '.join(risk_factors)}"
    
    evidence = {
        'fraud_probability': float(fraud_prob),
        'is_fraud': bool(is_fraud),
        'risk_level': calculate_risk_level(fraud_prob),
        'analyst_note': analyst_note,
        'top_risk_factors': [
            {
                'feature': feature,
                'value': float(value),
                'shap_contribution': float(shap_val),
                'impact': 'increases risk' if shap_val > 0 else 'decreases risk'
            }
            for feature, value, shap_val in top_drivers
        ]
    }
    
    return evidence


def save_to_database(transaction_id: str, transaction_data: Dict, evidence: Dict):
    """Save transaction and fraud alert to database"""
    try:
        # Save transaction
        transaction_record = {
            'transaction_id': transaction_id,
            'customer_id': transaction_data['customer_id'],
            'amount': transaction_data['amount'],
            'transaction_type': transaction_data['transaction_type'],
            'timestamp': datetime.now().isoformat(),
            'is_fraud': evidence['is_fraud'],
            'fraud_probability': evidence['fraud_probability'],
            'risk_level': evidence['risk_level']
        }
        
        supabase.table('transactions').insert(transaction_record).execute()
        
        # Save fraud alert if high risk
        if evidence['is_fraud'] or evidence['fraud_probability'] >= 0.4:
            alert_record = {
                'transaction_id': transaction_id,
                'fraud_probability': evidence['fraud_probability'],
                'analyst_note': evidence['analyst_note'],
                'risk_factors': evidence['top_risk_factors'],
                'status': 'pending'
            }
            supabase.table('fraud_alerts').insert(alert_record).execute()
        
        # Save multimodal features
        multimodal_record = {
            'transaction_id': transaction_id,
            'voice_liveness_score': transaction_data.get('voice_liveness_score'),
            'doc_anomaly_score': transaction_data.get('doc_anomaly_score'),
            'face_similarity': transaction_data.get('face_similarity'),
            'feature_vector': transaction_data
        }
        supabase.table('multimodal_features').insert(multimodal_record).execute()
        
        print(f"✅ Saved to database: {transaction_id}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# API Routes

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "SHAP-Guard Bank API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput, background_tasks: BackgroundTasks):
    """
    Predict fraud probability for a transaction
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate transaction ID
    transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
    
    # Convert input to dict
    features = transaction.dict()
    
    # Generate evidence card
    evidence = generate_evidence_card(features, model, explainer)
    
    # Save to database (async)
    background_tasks.add_task(save_to_database, transaction_id, features, evidence)
    
    # Return response
    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=evidence['fraud_probability'],
        is_fraud=evidence['is_fraud'],
        risk_level=evidence['risk_level'],
        analyst_note=evidence['analyst_note'],
        top_risk_factors=evidence['top_risk_factors'],
        timestamp=datetime.now().isoformat()
    )


@app.get("/alerts")
async def get_fraud_alerts(status: Optional[str] = None, limit: int = 50):
    """
    Get fraud alerts from database
    """
    try:
        query = supabase.table('fraud_alerts').select('*').order('created_at', desc=True).limit(limit)
        
        if status:
            query = query.eq('status', status)
        
        response = query.execute()
        
        return {
            "count": len(response.data),
            "alerts": response.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transactions")
async def get_transactions(limit: int = 50):
    """
    Get recent transactions
    """
    try:
        response = supabase.table('transactions').select('*').order('timestamp', desc=True).limit(limit).execute()
        
        return {
            "count": len(response.data),
            "transactions": response.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard-stats")
async def get_dashboard_stats():
    """
    Get statistics for dashboard
    """
    try:
        # Total transactions
        total_txn = supabase.table('transactions').select('id', count='exact').execute()
        
        # Fraud transactions
        fraud_txn = supabase.table('transactions').select('id', count='exact').eq('is_fraud', True).execute()
        
        # Pending alerts
        pending_alerts = supabase.table('fraud_alerts').select('id', count='exact').eq('status', 'pending').execute()
        
        # Recent high-risk transactions
        high_risk = supabase.table('transactions').select('*').eq('risk_level', 'HIGH').limit(10).execute()
        
        return {
            "total_transactions": total_txn.count,
            "fraud_detected": fraud_txn.count,
            "pending_alerts": pending_alerts.count,
            "fraud_rate": (fraud_txn.count / total_txn.count * 100) if total_txn.count > 0 else 0,
            "high_risk_transactions": high_risk.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/alerts/{alert_id}/review")
async def review_alert(alert_id: str, status: str, reviewed_by: str):
    """
    Update alert status after review
    """
    try:
        response = supabase.table('fraud_alerts').update({
            'status': status,
            'reviewed_by': reviewed_by,
            'reviewed_at': datetime.now().isoformat()
        }).eq('id', alert_id).execute()
        
        return {"message": "Alert updated successfully", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "database_connected": True,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 Starting SHAP-Guard Bank API Server")
    print("="*60)
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
