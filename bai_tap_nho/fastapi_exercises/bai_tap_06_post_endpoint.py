from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="POST Endpoint", description="Bài tập 06: POST endpoint với input model")

class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=1, description="Câu feedback")

@app.post("/predict")
async def predict_feedback(feedback: FeedbackInput):
    text = feedback.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    return {"original_text": text, "message": "Feedback received"}