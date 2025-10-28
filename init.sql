-- Transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_probability DECIMAL(5, 4),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Fraud alerts table
CREATE TABLE fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
    fraud_probability DECIMAL(5, 4) NOT NULL,
    analyst_note TEXT,
    risk_factors JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multimodal features table
CREATE TABLE multimodal_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
    voice_liveness_score DECIMAL(5, 4),
    doc_anomaly_score DECIMAL(5, 4),
    face_similarity DECIMAL(5, 4),
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX idx_fraud_alerts_created ON fraud_alerts(created_at);

-- Create view for dashboard
CREATE VIEW fraud_dashboard AS
SELECT 
    t.id,
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.transaction_type,
    t.timestamp,
    t.fraud_probability,
    t.risk_level,
    f.analyst_note,
    f.risk_factors,
    f.status,
    m.voice_liveness_score,
    m.doc_anomaly_score
FROM transactions t
LEFT JOIN fraud_alerts f ON t.transaction_id = f.transaction_id
LEFT JOIN multimodal_features m ON t.transaction_id = m.transaction_id
ORDER BY t.timestamp DESC;
