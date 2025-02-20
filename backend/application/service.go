package application

import "backend/domain"

// ModelService implements the PredictionService interface
type ModelService struct {
	GrpcClient domain.PredictionService
}

// NewModelService creates a new instance of ModelService
func NewModelService(grpcClient domain.PredictionService) *ModelService {
	return &ModelService{GrpcClient: grpcClient}
}

// Predict calls the gRPC client to make predictions
func (s *ModelService) Predict(input []float32) ([]float32, error) {
	return s.GrpcClient.Predict(input)
}
