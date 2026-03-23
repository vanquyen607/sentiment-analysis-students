from fastapi import FastAPI

app = FastAPI(title="Hello FastAPI", description="Bài tập 01: Ứng dụng FastAPI cơ bản")

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}