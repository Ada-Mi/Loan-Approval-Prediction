# Import libs

import unittest
import pickle
import numpy as np

class TestLoanApprovalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model and scaler
        with open("decisiontree.pkl", "rb") as f:
            cls.model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            cls.scaler = pickle.load(f)

    def test_model_loading(self):
        self.assertIsNotNone(self.model, "Model failed to load")
        self.assertIsNotNone(self.scaler, "Scaler failed to load")

    def test_prediction(self):
        # Sample test input
        sample_input = np.array([[50000, 750, 10000, 15, 1, 2, 0, 2, 1]])
        sample_input_scaled = self.scaler.transform(sample_input)
        
        # Ensure model predicts without errors
        prediction = self.model.predict(sample_input_scaled)
        probability = self.model.predict_proba(sample_input_scaled)[0][1]
        
        # Validate output format
        self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1")
        self.assertTrue(0.0 <= probability <= 1.0, "Probability should be between 0 and 1")


if __name__ == "__main__":
    unittest.main()
