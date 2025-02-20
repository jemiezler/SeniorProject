import grpc
from concurrent import futures
from src.proto import predict_pb2, predict_pb2_grpc
from src.application.service import PredictionService

# Load the ML model
prediction_service = PredictionService("src/model/lstm.h5")  # Ensure model exists

class PredictionServiceServicer(predict_pb2_grpc.PredictionServiceServicer):
    def Predict(self, request, context):
        """
        gRPC method to handle prediction requests.
        """
        features = list(request.features)  # Extract feature array from request
        predictions = prediction_service.predict(features)  # Run inference
        return predict_pb2.PredictionResponse(predictions=predictions)

def serve():
    """
    Starts the gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionServiceServicer(), server)
    server.add_insecure_port("[::]:50051")  # Exposing gRPC server on port 50051
    print("🚀 FastAPI gRPC Server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
