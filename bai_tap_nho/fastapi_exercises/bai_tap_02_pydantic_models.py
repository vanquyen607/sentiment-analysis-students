from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Pydantic Models", description="Bài tập 02: Sử dụng Pydantic models")

class Item(BaseModel):
    name: str = Field(..., min_length=1, description="Tên item")
    price: float = Field(..., gt=0, description="Giá item")
    is_offer: bool = Field(default=False, description="Có khuyến mãi không")

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item, "message": "Item created successfully"}