from pydantic import BaseModel

class CropPredictionRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropPredictionResponse(BaseModel):
    crop: str
    estimated_yield_per_hectare: float
    market_price_per_quintal: float
    estimated_profit_per_hectare: float
    feature_importances: dict
    actionable_suggestions: list[str]
