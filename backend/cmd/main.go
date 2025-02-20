package main

import (
	"log"

	"github.com/labstack/echo/v4"
	"backend/adapters/grpc"
	"backend/adapters/http"
	"backend/application"
)

func main() {
	e := echo.New()

	// Connect to gRPC Server (FastAPI)
	grpcClient, err := grpc.NewGRPCPredictionClient("localhost:50051") // Ensure FastAPI is running
	if err != nil {
		log.Fatalf("Failed to connect to gRPC server: %v", err)
	}
	defer grpcClient.Close()

	// Create Service
	service := application.NewModelService(grpcClient)

	// Set up Handlers
	handler := http.NewPredictionHandler(service)

	// Define Routes
	e.POST("/predict", handler.Predict)

	// Start Server
	e.Logger.Fatal(e.Start(":8080"))
}
