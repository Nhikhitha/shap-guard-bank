import pandas as pd
import numpy as np

class SyntheticTransactionGeneratorWithFlags:
    def __init__(self, n_samples=20000, fraud_ratio=0.05, random_state=42):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
        self.names = [f'C{i:07d}' for i in range(1, n_samples * 2)]  # customer ids

    def generate(self):
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_legit = self.n_samples - n_fraud

        df = pd.DataFrame({
            'step': np.random.randint(1, 744, size=self.n_samples),  # hourly steps in a month
            'type': np.random.choice(self.types, size=self.n_samples),
            'amount': np.round(np.random.exponential(scale=2000, size=self.n_samples), 2),
            'nameOrig': np.random.choice(self.names, size=self.n_samples),
            'oldbalanceOrg': np.round(np.random.uniform(0, 100000, size=self.n_samples), 2),
            'newbalanceOrig': 0.0,
            'nameDest': np.random.choice(self.names, size=self.n_samples),
            'oldbalanceDest': np.round(np.random.uniform(0, 100000, size=self.n_samples), 2),
            'newbalanceDest': 0.0,
            'isFraud': 0,
            'isFlaggedFraud': 0
        })

        df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
        df['newbalanceOrig'] = df['newbalanceOrig'].apply(lambda x: x if x > 0 else 0)
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']

        # Inject fraud cases
        fraud_indices = np.random.choice(df.index, n_fraud, replace=False)
        df.loc[fraud_indices, 'isFraud'] = 1
        df.loc[fraud_indices, 'amount'] *= np.random.uniform(2, 10, size=n_fraud)
        df.loc[fraud_indices, 'type'] = np.random.choice(['TRANSFER', 'CASH_OUT'], size=n_fraud)
        df.loc[fraud_indices, 'newbalanceOrig'] = 0

        # Rule-based flagged fraud: flag transactions for manual review based on:
        # - amount > 8000 or
        # - suspicious transaction type or
        # - large balance mismatch (e.g., newbalanceOrig + amount significantly different from oldbalanceOrg)
        flagged_conditions = (
            (df['amount'] > 8000) |
            (df['type'].isin(['TRANSFER', 'CASH_OUT'])) |
            (np.abs(df['oldbalanceOrg'] - (df['newbalanceOrig'] + df['amount'])) > 1000)
        )
        
        # Mark flagged fraud in both fraud and non-fraud transactions for suspicion
        df.loc[flagged_conditions, 'isFlaggedFraud'] = 1

        # Fraud likelihood scores to mimic multimodal cues
        df['fraud_likelihood_audio'] = 0.1
        df['fraud_likelihood_text'] = 0.1
        df.loc[fraud_indices, 'fraud_likelihood_audio'] = np.random.uniform(0.6, 1.0, size=n_fraud)
        df.loc[fraud_indices, 'fraud_likelihood_text'] = np.random.uniform(0.5, 1.0, size=n_fraud)

        df['amount'] = np.clip(df['amount'], 1, 100000)
        df['type_encoded'] = df['type'].map({k: v for v, k in enumerate(self.types)})

        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

def main():
    generator = SyntheticTransactionGeneratorWithFlags(n_samples=20000, fraud_ratio=0.05)
    df_synthetic = generator.generate()
    print(df_synthetic.head(10))
    df_synthetic.to_csv('synthetic_transactions_flagged.csv', index=False)

    print(f"Total transactions: {len(df_synthetic)}")
    print(f"Fraud transactions: {df_synthetic['isFraud'].sum()}")
    print(f"Flagged fraudsus: {df_synthetic['isFlaggedFraud'].sum()}")

main()
