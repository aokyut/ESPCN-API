from flask import Flask, request, jsonify
from networks import Espcn
import os
from io import BytesIO
import base64
from torchvision import transforms
from torch import load
import torch
import json
from PIL import Image

app = Flask(__name__)

device = "cpu"
model_path = "model/model_512000.pth"
net = Espcn(upscale=2)
net.load_state_dict(load(model_path, map_location="cpu"))
net.to(device)
net.eval()

def b64_to_PILImage(b64_string):
    """
    process convert base64 string to PIL Image
    input: b64_string: base64 string : data:image/png;base64,{base64 string}
    output: pil_img: PIL Image
    """
    b64_split = b64_string.split(",")[1]
    b64_bin = base64.b64decode(b64_split)
    with BytesIO(b64_bin) as b:
        pil_img = Image.open(b).copy().convert('RGB')
    return pil_img

def PILImage_to_b64(pil_img):
    """
    process convert PIL Image to base64
    input: pil_img: PIL Image
    output: b64_string: base64 string : data:image/png;base64,{base64 string}
    """
    with BytesIO() as b:
        pil_img.save(b, format='png')
        b.seek(0)
        img_base64 = base64.b64encode(b.read()).decode('utf-8')
    img_base64 = 'data:image/png;base64,' + img_base64
    return img_base64

def tensor_to_pil(src_tensor):
    src_tensor = src_tensor.mul(255)
    src_tensor = src_tensor.add_(0.5)
    src_tensor = src_tensor.clamp_(0, 255)
    src_tensor = src_tensor.permute(1, 2, 0)
    src_tensor = src_tensor.to("cpu", torch.uint8).numpy()
    return Image.fromarray(src_tensor)


def expand(src_image, model=net, device=device):
    src_tensor = transforms.ToTensor()(src_image).to(device)
    if src_tensor.dim() == 3:
        src_tensor = src_tensor.unsqueeze(0)
    
    srezo_tensor = model(src_tensor).squeeze()
    srezo_img = tensor_to_pil(srezo_tensor)
    return srezo_img

@app.route("/", methods=["POST"])
def superResolution():
    req_json = json.loads(request.data)
    src_b64 = req_json["srcImage"]

    # main process
    src_img = b64_to_PILImage(src_b64)
    srezo_img = expand(src_img)
    srezo_b64 = PILImage_to_b64(srezo_img)

    results = {"sresoImage":srezo_b64}

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 8000))