package domain

// Prediction represents the prediction result
type Prediction struct {
	Predictions []float32
}

// PredictionService defines the interface for prediction services
type PredictionService interface {
	Predict(features []float32) ([]float32, error)
}
