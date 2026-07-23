from dataclasses import dataclass
from typing import List

@dataclass
class ModelPerformance:
    """Data class to store model performance metrics"""
    model_name: str
    modality: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    confidence_score: float
    predicted_sections: List[str]
    true_sections: List[str] 