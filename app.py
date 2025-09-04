
from fastapi import FastAPI, UploadFile, File
import uvicorn
from inference import convert_voice

app = FastAPI()

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    output_path = convert_voice(await file.read())
    return {"converted": output_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
