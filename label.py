import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.io import decode_image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import base64
import torch
import os

# Convert tensor to bytes for response
import io
from PIL import Image

WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
print(WEIGHTS)

class RCNNAutoLabeler:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = fasterrcnn_resnet50_fpn(weights=WEIGHTS)
        self.categories = WEIGHTS.meta["categories"]
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x):
        return self.model(x)
    
    def get_class_name(self, label_id):
        return self.categories[label_id] if 0 <= label_id < len(self.categories) else f"unknown_{label_id}"

class ChessDataset(Dataset):
    def __init__(self, img_dir, auto_labeler=RCNNAutoLabeler(), size=500):
        self.img_dir = img_dir
        self.auto_labeler = auto_labeler
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize(size),
        ])

    def preprocess(self, img_path):
        img = decode_image(img_path)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = self.preprocess(img_path)
        predictions = self.auto_labeler.predict([image])
        label = predictions[0]
        return image, label
    
    def __len__(self):
        return len(self.image_files)


dataset = ChessDataset("./data")
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"

@app.get("/image/{index}")
async def get_image(index: int):
    if index < 0 or index >= len(dataset):
        return {"error": "Image not found"}

    img, label = dataset[index]

    # add label names with scores
    labels_names = [f"{dataset.auto_labeler.get_class_name(int(x[1]))}:{label['scores'][x[0]].item():.2f}" for x in enumerate(label['labels'])]

    # Convert to numpy and ensure correct data type for PIL
    img_array = img.permute(1, 2, 0).detach().cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue())

    return {"image": img_b64.decode(), 
            "bbox": label['boxes'].tolist(),
            "labels": label['labels'].tolist(),
            "label_names": labels_names,
            "scores": label['scores'].tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

