from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

data = preprocess_data()
model , X_test , y_test = train_model(data)
evaluate_model(model,X_test , y_test )