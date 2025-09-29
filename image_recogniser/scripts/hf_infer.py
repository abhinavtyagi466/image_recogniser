import os
import requests

HF_API = os.environ.get("HF_API_TOKEN")
if not HF_API:
    raise EnvironmentError("Set HF_API_TOKEN env var with your Hugging Face token.")

HEADERS = {"Authorization": f"Bearer {HF_API}"}

def image_caption_hf(image_path, model="Salesforce/blip-image-captioning-base"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(url, headers=HEADERS, data=data, timeout=60)
    if response.status_code == 200:
        out = response.json()
        if isinstance(out, list) and isinstance(out[0], dict):
            for key in ("generated_text", "caption", "text"):
                if key in out[0]:
                    return out[0][key]
        elif isinstance(out, dict):
            return out.get("generated_text") or out.get("caption") or str(out)
        return str(out)
    else:
        return f"HF error {response.status_code}: {response.text}"
