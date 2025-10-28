import numpy as np
import pandas as pd
import random
import uuid
import os
import soundfile as sf
from datetime import datetime, timedelta

# ========== 1️⃣ Generate Synthetic Audio Files ==========
def generate_synthetic_audio_file(folder="synthetic_audio", duration=2.0, sample_rate=16000):
    """
    Generate a synthetic .wav file with random sine tones + noise.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{uuid.uuid4()}.wav")

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * random.choice([440, 880, 1760]) * t)
    noise = np.random.normal(0, 0.2, tone.shape)
    audio = tone + noise

    sf.write(file_path, audio, sample_rate)
    return file_path


# ========== 2️⃣ Generate Synthetic Text or Audio Input ==========
def generate_audio_or_text_input():
    mode = random.choice(["audio", "text"])
    if mode == "audio":
        return {"mode": "audio", "file_path": generate_synthetic_audio_file()}
    else:
        texts = [
            "urgent transfer request",
            "verify your account immediately",
            "failed transaction alert",
            "your OTP is 985432",
            "confirm payment now",
            "successful purchase at electronics store",
            "password change alert"
        ]
        return {"mode": "text", "text": random.choice(texts)}


# ========== 3️⃣ Generate Synthetic Transaction Data ==========
def generate_transaction_data(num_records=10):
    accounts = ["Ravi Kumar", "Sneha Reddy", "Arjun Das", "Meena Iyer", "Karan Patel", "Priya Singh"]
    devices = ["Mobile", "ATM", "Web", "POS", "UPI"]
    locations = ["Bangalore", "Hyderabad", "Mumbai", "Delhi", "Chennai"]

    data = []
    for _ in range(num_records):
        tx_id = str(uuid.uuid4())[:8]
        holder = random.choice(accounts)
        device = random.choice(devices)
        amount = round(random.uniform(100, 50000), 2)
        avg_amount = round(amount * random.uniform(0.5, 1.5), 2)
        risk_score = round(random.uniform(0, 1), 2)
        location = random.choice(locations)
        tx_time = datetime.now() - timedelta(minutes=random.randint(0, 10000))

        # combine with audio/text input
        input_data = generate_audio_or_text_input()

        data.append({
            "transaction_id": tx_id,
            "account_holder": holder,
            "device": device,
            "amount": amount,
            "avg_amount": avg_amount,
            "risk_score": risk_score,
            "time": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "location": location,
            "input_type": input_data["mode"],
            "input_value": input_data.get("text", input_data.get("file_path"))
        })

    return pd.DataFrame(data)


# ========== 4️⃣ Run and Save the Data ==========
if __name__ == "__main__":
    df = generate_transaction_data(num_records=20)
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/synthetic_transactions.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Synthetic data generated successfully!")
    print(f"📁 Saved CSV: {output_path}")
    print(f"🎧 Audio files saved in: synthetic_audio/")
    print("\nSample Preview:\n", df.head(5))
