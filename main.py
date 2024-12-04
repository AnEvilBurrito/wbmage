from app import load_test_images, safe_read_image, standardize_image, process_image, detect_bands, analyze_band_intensity 


from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/load_test_images/{image_path}")
def get_images(image_path: str):
    return load_test_images(image_path)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
