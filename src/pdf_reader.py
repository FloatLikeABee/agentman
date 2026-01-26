"""
PDF Reader Tool
Reads PDF files and processes content with AI based on system prompt
"""
import logging
import io
from typing import Dict, Any, Optional
from datetime import datetime

from .config import settings
from .llm_factory import LLMFactory, LLMProvider
from .models import LLMProviderType


class PDFReader:
    """PDF reader with AI processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def _extract_text_from_pdf(self, pdf_data: bytes) -> Dict[str, Any]:
        """
        Extract text from PDF file
        
        Args:
            pdf_data: PDF file bytes
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Try pypdf first (newer library)
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_data))
                pages = []
                for page in pdf_reader.pages:
                    text = page.extract_text() or ''
                    pages.append(text)
                
                full_text = "\n".join(pages)
                return {
                    "success": True,
                    "text": full_text,
                    "page_count": len(pdf_reader.pages),
                    "library": "pypdf"
                }
            except ImportError:
                # Fallback to PyPDF2
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                    pages = []
                    for page in pdf_reader.pages:
                        text = page.extract_text() or ''
                        pages.append(text)
                    
                    full_text = "\n".join(pages)
                    return {
                        "success": True,
                        "text": full_text,
                        "page_count": len(pdf_reader.pages),
                        "library": "PyPDF2"
                    }
                except ImportError:
                    return {
                        "success": False,
                        "error": "PDF support requires 'pypdf' or 'PyPDF2' package. Install with: pip install pypdf"
                    }
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return {
                "success": False,
                "error": f"Error reading PDF: {str(e)}"
            }
    
    def _process_with_ai(
        self,
        text: str,
        system_prompt: str,
        llm_provider: Optional[LLMProviderType] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process extracted text with AI based on system prompt
        
        Args:
            text: Extracted text from PDF
            system_prompt: System prompt for AI processing
            llm_provider: Optional LLM provider (default: from settings)
            model_name: Optional model name (default: provider default)
            
        Returns:
            Dictionary with AI-processed result
        """
        try:
            # Determine provider and model
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
            else:
                provider = LLMProvider.GEMINI
                api_key = settings.gemini_api_key
                model = settings.gemini_default_model
            
            if not api_key:
                return {
                    "success": False,
                    "error": f"{provider_str.capitalize()} API key not configured"
                }
            
            # Create LLM caller
            llm_caller = LLMFactory.create_caller(
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=0.7,
                max_tokens=8192
            )
            
            # Build the prompt
            full_prompt = f"{system_prompt}\n\nPDF Content:\n{text}"
            
            # Generate response
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
        pdf_data: bytes,
        system_prompt: str,
        llm_provider: Optional[LLMProviderType] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read PDF and process with AI
        
        Args:
            pdf_data: PDF file bytes
            system_prompt: System prompt for AI processing
            llm_provider: Optional LLM provider
            model_name: Optional model name
            
        Returns:
            Dictionary with extracted text, AI result, and metadata
        """
        try:
            # Extract text from PDF
            extraction_result = self._extract_text_from_pdf(pdf_data)
            
            if not extraction_result.get("success"):
                return extraction_result
            
            extracted_text = extraction_result["text"]
            page_count = extraction_result.get("page_count", 0)
            
            # Check if text is empty
            if not extracted_text or not extracted_text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from the PDF. The PDF might be image-based or corrupted.",
                    "page_count": page_count
                }
            
            # Process with AI
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
                    "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                    "page_count": page_count
                }
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "extracted_text_length": len(extracted_text),
                "page_count": page_count,
                "ai_result": ai_result["result"],
                "provider": ai_result["provider"],
                "model": ai_result["model"],
                "system_prompt": system_prompt,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error reading and processing PDF: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
