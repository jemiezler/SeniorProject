package http

import (
	"net/http"

	"github.com/labstack/echo/v4"
	"backend/application"
)

type PredictionHandler struct {
	Service *application.ModelService
}

// NewPredictionHandler creates a new HTTP handler for predictions
func NewPredictionHandler(service *application.ModelService) *PredictionHandler {
	return &PredictionHandler{Service: service}
}

// HTTP handler for prediction
func (h *PredictionHandler) Predict(c echo.Context) error {
	var input struct {
		Features []float32 `json:"features"`
	}

	if err := c.Bind(&input); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid input"})
	}

	result, err := h.Service.Predict(input.Features)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "Prediction failed"})
	}

	return c.JSON(http.StatusOK, map[string]interface{}{"prediction": result})
}
