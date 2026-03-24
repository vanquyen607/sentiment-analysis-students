from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="File Response", description="Bài tập 05: Serve HTML với FileResponse")

# Tạo thư mục static
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Tạo file index.html đơn giản
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Frontend</title>
</head>
<body>
    <h1>Hello from FastAPI!</h1>
    <p>This is a simple HTML page served by FastAPI.</p>
</body>
</html>
"""
(static_dir / "index.html").write_text(html_content)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')