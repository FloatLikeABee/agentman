"""
Image Reader Tool using Qwen Vision OCR Model
Reads text and extracts information from images using Qwen's vision model
"""
import logging
import base64
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image

from .config import settings


class ImageReader:
    """Image reader using Qwen Vision OCR model"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.qwen_api_key
        self.base_url = settings.qwen_base_url
        self.model = "qwen-vl-ocr-2025-11-20"
        
        if not self.api_key:
            self.logger.warning("Qwen API key not configured. Image reader will not work.")
    
    def _image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _get_image_data_url(self, image_data: bytes, mime_type: str = "image/jpeg") -> str:
        """Convert image bytes to data URL"""
        base64_str = self._image_to_base64(image_data)
        return f"data:{mime_type};base64,{base64_str}"
    
    def _get_image_info(self, image_data: bytes) -> Dict[str, Any]:
        """Get image dimensions and format"""
        try:
            img = Image.open(BytesIO(image_data))
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
        except Exception as e:
            self.logger.error(f"Error getting image info: {e}")
            return {}
    
    def read_image(
        self, 
        image_data: bytes, 
        prompt: Optional[str] = None,
        min_pixels: int = 32 * 32 * 3,
        max_pixels: int = 32 * 32 * 8192
    ) -> Dict[str, Any]:
        """
        Read text from an image using Qwen Vision OCR model
        
        Args:
            image_data: Image file bytes
            prompt: Optional custom prompt (default: OCR extraction prompt)
            min_pixels: Minimum pixel threshold for image scaling
            max_pixels: Maximum pixel threshold for image scaling
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "Qwen API key not configured"
            }
        
        try:
            # Default prompt for OCR extraction
            default_prompt = "Please output only the text content from the image without any additional descriptions or formatting."
            text_prompt = prompt or default_prompt
            
            # Convert image to base64 data URL
            image_data_url = self._get_image_data_url(image_data)
            
            # Get image info
            image_info = self._get_image_info(image_data)
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                },
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels
                            },
                            {
                                "type": "text",
                                "text": text_prompt
                            }
                        ]
                    }
                ]
            }
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Qwen API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
            
            result = response.json()
            
            # Extract text from response
            extracted_text = ""
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    extracted_text = choice["message"]["content"]
            
            return {
                "success": True,
                "text": extracted_text,
                "image_info": image_info,
                "model": self.model,
                "prompt_used": text_prompt,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling Qwen API: {e}")
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Error reading image: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def read_multiple_images(
        self,
        images_data: List[bytes],
        prompt: Optional[str] = None,
        min_pixels: int = 32 * 32 * 3,
        max_pixels: int = 32 * 32 * 8192
    ) -> Dict[str, Any]:
        """
        Read text from multiple images sequentially
        
        Args:
            images_data: List of image file bytes
            prompt: Optional custom prompt
            min_pixels: Minimum pixel threshold
            max_pixels: Maximum pixel threshold
            
        Returns:
            Dictionary with results for each image
        """
        results = []
        
        for idx, image_data in enumerate(images_data):
            self.logger.info(f"Processing image {idx + 1} of {len(images_data)}")
            result = self.read_image(image_data, prompt, min_pixels, max_pixels)
            result["image_index"] = idx + 1
            results.append(result)
        
        return {
            "success": True,
            "total_images": len(images_data),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
