import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template, send_file
from PIL import Image
import os

from models.generator import Generator

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = Generator().to(device)
model.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    path = "uploads/input.png"
    file.save(path)

    # load image
    img = Image.open(path).convert("L")
    img = transform(img).unsqueeze(0).to(device)

    # predict
    with torch.no_grad():
        fake_rgb = model(img)

    output_path = "outputs/result.png"
    transforms.ToPILImage()(fake_rgb.squeeze(0).cpu()).save(output_path)

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)