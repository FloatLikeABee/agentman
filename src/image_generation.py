"""
Shared image generation and save logic for the Images API and Graphic Document Generator.
Uses Pollinations API to generate images and saves them under data/generated_images.
"""
import os
import json
import uuid
import requests
from urllib.parse import quote_plus
from datetime import datetime


def get_images_dir():
    """Return the directory where generated images are stored (for API file serving)."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_images')


def _get_images_dir():
    return get_images_dir()


def _get_metadata_file():
    return os.path.join(_get_images_dir(), 'images_metadata.json')


def load_images_metadata():
    path = _get_metadata_file()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_images_metadata(metadata):
    path = _get_metadata_file()
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_image_and_save(prompt: str, save: bool = True) -> dict:
    """
    Generate an image from a text prompt using Pollinations API.
    If save=True, download and save to data/generated_images and update metadata.
    Returns dict with: image_url, prompt, saved, filename (if saved), save_error (if save failed).
    """
    if not prompt or not prompt.strip():
        return {"image_url": "", "prompt": prompt, "saved": False, "save_error": "Prompt is required"}
    prompt = prompt.strip()
    encoded_prompt = quote_plus(prompt)
    image_url = f"https://gen.pollinations.ai/image/{encoded_prompt}?model=flux"
    result = {"image_url": image_url, "prompt": prompt, "saved": False}
    if not save:
        return result
    images_dir = _get_images_dir()
    os.makedirs(images_dir, exist_ok=True)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"img_{timestamp}_{unique_id}.png"
        filepath = os.path.join(images_dir, filename)
        headers = {
            'Authorization': 'Bearer pk_gzM6GGeMB1HQWKgW',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        response = requests.get(image_url, headers=headers, timeout=60)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            metadata = load_images_metadata()
            metadata.insert(0, {
                "filename": filename,
                "prompt": prompt,
                "created_at": datetime.now().isoformat(),
                "url": image_url,
            })
            save_images_metadata(metadata)
            result["saved"] = True
            result["filename"] = filename
        else:
            result["save_error"] = f"Failed to download image: {response.status_code}"
    except Exception as e:
        result["save_error"] = str(e)
    return result
