from fastapi import FastAPI, File, UploadFile, Query,Form
from application.services import AnalysisService

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), temp: float = Form(...)):
    """API to accept image and temperature, then predict class using extracted features."""
    try:
        image_bytes = await file.read()
        result = AnalysisService.analyze_image(image_bytes, temp)
        return result
    except ValueError as e:
        return {"error": str(e)}

#('Lab', 'HSV', 'GLCM', 'LBP', 'Temp', 'Yellow', 'Cyan', 'Magenta', 'Brightness', 'Chroma')
