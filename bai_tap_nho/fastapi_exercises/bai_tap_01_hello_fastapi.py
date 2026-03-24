from fastapi import FastAPI

app = FastAPI(title="Hello FastAPI", description="Bài tập 01: Ứng dụng FastAPI cơ bản")

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

# uvicorn fastapi_exercises.bai_tap_01_hello_fastapi:app --host 0.0.0.0 --port 8000