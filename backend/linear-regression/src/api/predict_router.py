from fastapi import APIRouter, File, UploadFile, Form
from src.application.services import AnalysisService
from fastapi.responses import Response, StreamingResponse

router = APIRouter()

@router.post("/predict/")
async def predict(
    file: UploadFile = File(...), 
    temp: float = Form(...)
):
    """
    API to accept an image and temperature, then predict output using extracted features.
    """
    try:
        # Read image bytes
        image = await file.read()
        prediction = AnalysisService.predict_image(image, temp)
        
        return {
            "status": "success",
            "data": prediction
        }


    except ValueError as e:
        return {"status": "error", "message": str(e)}

    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}
