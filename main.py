import base64
from io import BytesIO

import requests
from diffusers.utils import load_image
from PIL import Image

def encode_image(img):
    "Encode PIL.Image into base64 string"
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string

def decode_image(encoded_string):
    "Decode base64 string into PIL.Image"
    image_data = base64.b64decode(encoded_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

image = load_image(
    "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
)

token = HYPERBOLIC_API_KEY

headers = {
    'Authorization': f'Bearer {token}',
}


result = requests.post(
    url="https://api.hyperbolic.xyz/v1/image/generation",
    headers=headers,
    json={
        "prompt": "an astronaut on Mars",
        "backend": "auto",
        "model_name": "SDXL-ControlNet",
        "height": 1024,
        "width": 1024,
        "controlnet_name": "depth",
        "controlnet_image": encode_image(image),
        "seed": 5742320,
        "cfg_scale": 15,
    },
)

image_str = result.json()["images"][0]["image"]

img = decode_image(image_str)
img.save("result.png")

# just to check git merge