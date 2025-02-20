package grpc

import (
	"context"
	"log"
	"time"

	"backend/domain"
	pb "backend/proto"

	"google.golang.org/grpc"
)

type GRPCPredictionClient struct {
	conn   *grpc.ClientConn
	client pb.PredictionServiceClient
}

// NewGRPCPredictionClient establishes a gRPC connection to the FastAPI server.
func NewGRPCPredictionClient(serverAddr string) (*GRPCPredictionClient, error) {
	conn, err := grpc.Dial(serverAddr, grpc.WithInsecure(), grpc.WithBlock()) // Use TLS in production
	if err != nil {
		return nil, err
	}

	client := pb.NewPredictionServiceClient(conn)
	return &GRPCPredictionClient{conn: conn, client: client}, nil
}

// Predict sends a feature vector to the FastAPI gRPC server and returns predictions.
func (c *GRPCPredictionClient) Predict(features []float32) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	req := &pb.PredictionRequest{Features: features}
	res, err := c.client.Predict(ctx, req)
	if err != nil {
		return nil, err
	}

	return res.Predictions, nil
}

// Close closes the gRPC connection
func (c *GRPCPredictionClient) Close() {
	c.conn.Close()
}
