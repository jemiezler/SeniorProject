from fastapi import FastAPI
from src.api import predict_router
app = FastAPI(title="Image Feature Extraction & Prediction API")

# Include API routes
app.include_router(predict_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, debug=True)
