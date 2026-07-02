import os
import time
import numpy as np
from typing import Dict, Any, Union


class Predictor:
    """Base interface for model prediction backends."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SklearnPredictor(Predictor):
    """Predictor using standard scikit-learn via pickle."""
    
    def __init__(self, model_path: str):
        # Local import to avoid contaminating ONNX edge cases
        import pickle
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class OnnxPredictor(Predictor):
    """Predictor using onnxruntime.
    
    WARNING: This class must NEVER import sklearn, scipy, or joblib.
    """
    
    def __init__(self, model_path: str):
        # Lazy import of onnxruntime
        import onnxruntime as rt
        
        # We suppress warnings/logging to keep output clean, but allow basic optimizations
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = rt.InferenceSession(model_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name
        # The RandomForest in ONNX usually returns [labels, probabilities]
        self.label_name = self.session.get_outputs()[0].name
        self.proba_name = self.session.get_outputs()[1].name
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        outputs = self.session.run([self.label_name], {self.input_name: X})
        return outputs[0]
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        outputs = self.session.run([self.proba_name], {self.input_name: X})
        # outputs[1] is a list of dicts: [{0: p0, 1: p1}, {0: p0, 1: p1}, ...]
        # We need to convert this to an (N, 2) array
        probas = outputs[0]
        result = np.zeros((len(probas), 2), dtype=np.float32)
        for i, p_dict in enumerate(probas):
            result[i, 0] = p_dict.get(0, 0.0)
            result[i, 1] = p_dict.get(1, 0.0)
        return result


def convert_to_onnx(sklearn_model, n_features: int, output_path: str):
    """Convert a trained scikit-learn model to ONNX format.
    
    Must only be called during training/preprocessing, never during edge inference.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Enable zipmap=False for better performance if possible, but we'll stick to default 
    # to ensure it works correctly with dict outputs as handled in predict_proba.
    onx = convert_sklearn(sklearn_model, initial_types=initial_type, target_opset=12)
    
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())


def get_predictor(backend: str, model_path: str) -> Predictor:
    """Factory to get the requested predictor.
    
    Args:
        backend: 'sklearn' or 'onnx'
        model_path: Path to the .pkl or .onnx file
    """
    if backend == 'onnx':
        return OnnxPredictor(model_path)
    elif backend == 'sklearn':
        return SklearnPredictor(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'onnx' or 'sklearn'.")
