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
from .llm_factory import LLMFactory, LLMProvider
from .models import LLMProviderType


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

    def _process_with_ai(
        self,
        text: str,
        system_prompt: str,
        llm_provider: Optional[LLMProviderType] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process extracted text with AI based on system prompt.

        Args:
            text: Extracted text from image(s)
            system_prompt: System prompt for AI processing
            llm_provider: Optional LLM provider (default: from settings)
            model_name: Optional model name (default: provider default)

        Returns:
            Dictionary with AI-processed result
        """
        try:
            provider_str = llm_provider.value if llm_provider else settings.default_llm_provider.lower()

            if provider_str == "gemini":
                provider = LLMProvider.GEMINI
                api_key = settings.gemini_api_key
                model = model_name or settings.gemini_default_model
            elif provider_str == "qwen":
                provider = LLMProvider.QWEN
                api_key = settings.qwen_api_key
                model = model_name or settings.qwen_default_model
            elif provider_str == "mistral":
                provider = LLMProvider.MISTRAL
                api_key = settings.mistral_api_key
                model = model_name or settings.mistral_default_model
            elif provider_str == "groq":
                provider = LLMProvider.GROQ
                api_key = getattr(settings, "groq_api_key", "")
                model = model_name or getattr(settings, "groq_default_model", "llama-3.3-70b-versatile")
            else:
                provider = LLMProvider.GEMINI
                api_key = settings.gemini_api_key
                model = settings.gemini_default_model

            if not api_key:
                return {
                    "success": False,
                    "error": f"{provider_str.capitalize()} API key not configured"
                }

            llm_caller = LLMFactory.create_caller(
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=0.7,
                max_tokens=8192
            )

            full_prompt = f"{system_prompt}\n\nImage content (extracted text):\n{text}"
            result = llm_caller.generate(full_prompt)

            return {
                "success": True,
                "result": result,
                "provider": provider_str,
                "model": model,
                "system_prompt": system_prompt
            }
        except Exception as e:
            self.logger.error(f"Error processing text with AI: {e}")
            return {
                "success": False,
                "error": f"Error processing with AI: {str(e)}"
            }

    def read_and_process(
        self,
        image_data: bytes,
        system_prompt: str,
        llm_provider: Optional[LLMProviderType] = None,
        model_name: Optional[str] = None,
        ocr_prompt: Optional[str] = None,
        min_pixels: int = 32 * 32 * 3,
        max_pixels: int = 32 * 32 * 8192
    ) -> Dict[str, Any]:
        """
        Read text from image using OCR, then process with AI using the given system prompt.

        Args:
            image_data: Image file bytes
            system_prompt: System prompt for AI processing of the extracted content
            llm_provider: Optional LLM provider for AI step
            model_name: Optional model name for AI step
            ocr_prompt: Optional custom prompt for OCR (default: extract text only)
            min_pixels: Minimum pixel threshold for OCR
            max_pixels: Maximum pixel threshold for OCR

        Returns:
            Dictionary with extracted text, AI result, and metadata
        """
        try:
            read_result = self.read_image(
                image_data=image_data,
                prompt=ocr_prompt,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )

            if not read_result.get("success"):
                return read_result

            extracted_text = read_result.get("text", "").strip()
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No text could be extracted from the image.",
                    "image_info": read_result.get("image_info"),
                    "timestamp": datetime.now().isoformat()
                }

            ai_result = self._process_with_ai(
                text=extracted_text,
                system_prompt=system_prompt,
                llm_provider=llm_provider,
                model_name=model_name
            )

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "error": ai_result.get("error", "Failed to process with AI"),
                    "extracted_text": extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""),
                    "image_info": read_result.get("image_info"),
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "success": True,
                "extracted_text": extracted_text,
                "extracted_text_length": len(extracted_text),
                "image_info": read_result.get("image_info"),
                "ai_result": ai_result["result"],
                "provider": ai_result["provider"],
                "model": ai_result["model"],
                "system_prompt": system_prompt,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error in read_and_process: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def read_and_process_multi(
        self,
        images_data: list,
        system_prompt: str,
        llm_provider: Optional[LLMProviderType] = None,
        model_name: Optional[str] = None,
        ocr_prompt: Optional[str] = None,
        min_pixels: int = 32 * 32 * 3,
        max_pixels: int = 32 * 32 * 8192
    ) -> Dict[str, Any]:
        """
        Read text from multiple images (OCR each), concatenate, then process with AI once.

        Args:
            images_data: List of image file bytes
            system_prompt: System prompt for AI processing of the combined content
            llm_provider: Optional LLM provider for AI step
            model_name: Optional model name for AI step
            ocr_prompt: Optional custom prompt for OCR (used for all images)
            min_pixels: Minimum pixel threshold for OCR
            max_pixels: Maximum pixel threshold for OCR

        Returns:
            Same shape as read_and_process: extracted_text (combined), ai_result, etc.
        """
        if not images_data:
            return {
                "success": False,
                "error": "No images provided.",
                "timestamp": datetime.now().isoformat()
            }
        try:
            texts = []
            for i, image_data in enumerate(images_data):
                read_result = self.read_image(
                    image_data=image_data,
                    prompt=ocr_prompt,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                if read_result.get("success"):
                    t = (read_result.get("text") or "").strip()
                    if t:
                        texts.append(t)
                # On failure we still continue with other images; optional: collect error per image

            combined_text = "\n\n".join(texts).strip()
            if not combined_text:
                return {
                    "success": False,
                    "error": "No text could be extracted from any of the images.",
                    "timestamp": datetime.now().isoformat()
                }

            ai_result = self._process_with_ai(
                text=combined_text,
                system_prompt=system_prompt,
                llm_provider=llm_provider,
                model_name=model_name
            )

            if not ai_result.get("success"):
                return {
                    "success": False,
                    "error": ai_result.get("error", "Failed to process with AI"),
                    "extracted_text": combined_text[:500] + ("..." if len(combined_text) > 500 else ""),
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "success": True,
                "extracted_text": combined_text,
                "extracted_text_length": len(combined_text),
                "image_count": len(images_data),
                "ai_result": ai_result["result"],
                "provider": ai_result["provider"],
                "model": ai_result["model"],
                "system_prompt": system_prompt,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error in read_and_process_multi: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
