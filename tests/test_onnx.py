import os
import pytest
import numpy as np
from src.classifier import train_random_forest
from src.onnx_model import convert_to_onnx, get_predictor

@pytest.fixture
def mock_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_onnx_conversion_and_parity(tmp_path, mock_data):
    X, y = mock_data
    
    # 1. Train Sklearn Model
    sklearn_model = train_random_forest(X, y, n_estimators=10)
    
    # 2. Convert to ONNX
    onnx_path = os.path.join(tmp_path, "model.onnx")
    convert_to_onnx(sklearn_model, n_features=5, output_path=onnx_path)
    
    assert os.path.exists(onnx_path)
    
    # 3. Load via Predictors
    # We cheat slightly in the test and load the pickle manually for the test comparison
    import pickle
    pkl_path = os.path.join(tmp_path, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(sklearn_model, f)
        
    sk_predictor = get_predictor("sklearn", pkl_path)
    onnx_predictor = get_predictor("onnx", onnx_path)
    
    # 4. Assert Parity
    sk_preds = sk_predictor.predict(X)
    onnx_preds = onnx_predictor.predict(X)
    
    np.testing.assert_array_equal(sk_preds, onnx_preds, err_msg="ONNX and Sklearn predictions mismatch")
    
    sk_probas = sk_predictor.predict_proba(X)
    onnx_probas = onnx_predictor.predict_proba(X)
    
    np.testing.assert_allclose(sk_probas, onnx_probas, rtol=1e-3, atol=1e-4, 
                               err_msg="ONNX and Sklearn probabilities mismatch")
