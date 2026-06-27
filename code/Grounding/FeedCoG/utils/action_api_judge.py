"""
API-based Judge for Two-Box Selection
======================================
Supports:
1. OpenAI-compatible APIs (GPT-4o, Claude, etc.)
2. Google Gemini API (including thinking models like Gemini 2.5 Pro)

Usage:
    # Standard API (OpenAI-compatible)
    judge = APIJudgeTwoImages(
        api_type="openai",
        api_key="<OPENAI_API_KEY>",
        base_url="https://api.openai.com/v1",
        model="gpt-4o"
    )
    
    # Gemini API (standard)
    judge = APIJudgeTwoImages(
        api_type="gemini",
        api_key="<GEMINI_API_KEY>",
        model="gemini-2.0-flash"
    )
    
    # Gemini Thinking Model
    judge = APIJudgeTwoImages(
        api_type="gemini_thinking",
        api_key="<GEMINI_API_KEY>",
        model="gemini-2.5-pro-preview-05-06",
        thinking_budget=8192  # Optional thinking token budget
    )
"""

import os
import re
import json
import base64
import logging
from io import BytesIO
from copy import deepcopy
from typing import Tuple, Optional

from PIL import Image

from .action import BaseAction


class APIJudgeTwoImages(BaseAction):
    """
    API-based judge supporting multiple backends:
    - openai: OpenAI-compatible APIs (OpenAI, Azure, OpenRouter, etc.)
    - gemini: Google Gemini standard models
    - gemini_thinking: Google Gemini thinking models (e.g., Gemini 2.5 Pro)
    """
    _required_keys_ = ["image", "image1", "image2"]
    
    def __init__(
        self,
        api_type: str = "openai",
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        thinking_budget: int = None,
        max_tokens: int = None,
        site_url: str = None,
        site_title: str = None,
        max_retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize API Judge
        
        Args:
            api_type: "openai", "gemini", or "gemini_thinking"
            api_key: API key (can also use env vars)
            base_url: API base URL (for OpenAI-compatible)
            model: Model name
            thinking_budget: Max thinking tokens (for gemini_thinking)
            max_tokens: Max output tokens (default 16384 for thinking models, 2048 for others)
            site_url: Site URL for OpenRouter
            site_title: Site title for OpenRouter
            max_retries: Max retry attempts
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_type = api_type.lower()
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking_budget = thinking_budget
        
        # Set max_tokens based on model type (thinking models need more)
        if max_tokens:
            self.max_tokens = max_tokens
        elif model and ("thinking" in model.lower() or "gemini-3" in model.lower() or "2.5-pro" in model.lower()):
            self.max_tokens = 16384  # Thinking models need more tokens
        else:
            self.max_tokens = 4096  # Default for regular models
        
        # Resolve API key from environment
        if api_key:
            self.api_key = api_key
        elif self.api_type.startswith("gemini"):
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        else:
            self.api_key = (
                os.environ.get("OPENROUTER_API_KEY") or 
                os.environ.get("OPENAI_API_KEY")
            )
        
        if not self.api_key:
            raise ValueError(f"API key not provided for {api_type}")
        
        # Set defaults based on api_type
        if self.api_type == "openai":
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            self.model = model or "gpt-4o"
        elif self.api_type == "gemini":
            self.model = model or "gemini-2.0-flash"
        elif self.api_type == "gemini_thinking":
            self.model = model or "gemini-2.5-pro-preview-05-06"
            self.thinking_budget = thinking_budget or 8192
        else:
            raise ValueError(f"Unknown api_type: {api_type}. Use 'openai', 'gemini', or 'gemini_thinking'")
        
        self.base_url = base_url
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.site_title = site_title or os.environ.get("OPENROUTER_SITE_TITLE") or "GUI-Agent"
        
        logging.info(f"APIJudgeTwoImages initialized: type={self.api_type}, model={self.model}, max_tokens={self.max_tokens}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _build_prompt(self, user_query: str) -> str:
        """Build the judgment prompt"""
        return f"""You are comparing two images to determine which one better fulfills the user's intent.

User Command: "{user_query}"

Image 1: Shows a GUI element marked with a green box labeled "1"
Image 2: Shows a GUI element marked with a red box labeled "2"

Your task: Determine which image shows the element that will best fulfill the user's command.

ANALYSIS APPROACH:
1. Examine what GUI element is highlighted in each image
2. Consider which element better matches the user's intent
3. Think about standard GUI patterns and user expectations
4. Choose the image that shows the more appropriate interaction target

KEY PRINCIPLES:
- Focus on the functional purpose of the highlighted elements
- Consider standard UI patterns (buttons for actions, text fields for input, etc.)
- Choose interactive elements over static text/labels
- If one shows a selected state and the other shows normal state, prefer the normal state
- ELEMENT QUALITY HIERARCHY (best to worst):
   - Icon + Text together (most informative and complete)
   - Complete icon alone (clear visual indicator)  
   - Complete text alone (readable label)
   - Multiple elements in one box OR incomplete elements (ambiguous target)

COMMON PITFALLS TO AVOID:
    - Don't choose based on keyword matching alone
    - Don't overlook the user's actual goal in favor of literal interpretation

Remember: Provide SPECIFIC analysis based on what you actually observe, not generic descriptions.

**OUTPUT FORMAT**:
<analysis>
Image 1: [Describe what element is highlighted and its purpose]
Image 2: [Describe what element is highlighted and its purpose]
Comparison: [Explain which better serves the user's intent and why]
</analysis>

<answer>1 or 2</answer>
<reason>Brief explanation of why this image shows the better choice</reason>"""

    def _call_openai_api(
        self, 
        image1: Image.Image, 
        image2: Image.Image, 
        user_query: str
    ) -> Tuple[str, str, str]:
        """Call OpenAI-compatible API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        
        img1_base64 = self._image_to_base64(image1)
        img2_base64 = self._image_to_base64(image2)
        prompt = self._build_prompt(user_query)
        
        extra_headers = None
        if self.base_url and "openrouter.ai" in self.base_url:
            extra_headers = {
                "HTTP-Referer": self.site_url or "https://localhost",
                "X-Title": self.site_title,
            }
        
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    extra_headers=extra_headers,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img1_base64}",
                                        "detail": "high"
                                    }
                                },
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img2_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens
                )
                
                # Extract response text with proper null handling
                response_text = ""
                if response.choices and len(response.choices) > 0:
                    msg = response.choices[0].message
                    if msg.content:
                        response_text = msg.content
                    # Some APIs put content in different fields
                    elif hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                        response_text = msg.reasoning_content
                    # Try to extract from tool_calls if present
                    elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                        response_text = str(msg.tool_calls)
                    
                    # Log full message structure for debugging
                    logging.info(f"[DEBUG] Message object: content={msg.content is not None}, "
                                f"has_reasoning={hasattr(msg, 'reasoning_content')}")
                
                if not response_text:
                    # Log full response structure for debugging
                    logging.warning(f"[DEBUG] Empty response! Inspecting full response object...")
                    logging.warning(f"[DEBUG] response.choices: {response.choices}")
                    if response.choices:
                        choice = response.choices[0]
                        logging.warning(f"[DEBUG] choice.message: {choice.message}")
                        logging.warning(f"[DEBUG] choice dir: {[a for a in dir(choice) if not a.startswith('_')]}")
                        logging.warning(f"[DEBUG] message dir: {[a for a in dir(choice.message) if not a.startswith('_')]}")
                    response_text = ""
                
                return self._parse_response(response_text)
                
            except Exception as e:
                logging.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return "1", "API call failed", "error"

    def _call_gemini_api(
        self, 
        image1: Image.Image, 
        image2: Image.Image, 
        user_query: str,
        use_thinking: bool = False
    ) -> Tuple[str, str, str]:
        """Call Google Gemini API"""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")
        
        client = genai.Client(api_key=self.api_key)
        prompt = self._build_prompt(user_query)
        
        # Prepare content parts
        contents = [
            types.Part.from_text(prompt),
            types.Part.from_image(image1),
            types.Part.from_image(image2)
        ]
        
        for attempt in range(self.max_retries):
            try:
                if use_thinking:
                    # Gemini Thinking Model configuration
                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        ),
                        temperature=0
                    )
                    response = client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                else:
                    # Standard Gemini
                    config = types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=2048
                    )
                    response = client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                
                # Extract response text
                response_text = ""
                thinking_text = ""
                
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        thinking_text += part.text + "\n"
                    elif hasattr(part, 'text'):
                        response_text += part.text
                
                if thinking_text:
                    logging.info(f"Gemini thinking process:\n{thinking_text[:500]}...")
                
                return self._parse_response(response_text, thinking_text)
                
            except Exception as e:
                logging.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return "1", "API call failed", "error"

    def _parse_response(
        self, 
        response_text: str, 
        thinking_text: str = ""
    ) -> Tuple[str, str, str]:
        """Parse API response to extract selection and reason"""
        # Handle None or empty response
        if not response_text:
            logging.warning("[DEBUG] Response text is empty or None!")
            return "1", "Empty API response - defaulting to Image 1", ""
        
        logging.info(f"[DEBUG] Raw API response (len={len(response_text)}):\n{response_text[:1500]}")
        
        selected = None
        reason = None
        
        # Method 1: Try XML format (<analysis>, <answer>, <reason>)
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(\d)\s*</answer>', response_text)
        reason_match = re.search(r'<reason>(.*?)</reason>', response_text, re.DOTALL)
        
        if answer_match:
            selected = answer_match.group(1)
        if reason_match:
            reason = reason_match.group(1).strip()
        if analysis_match and reason:
            analysis = analysis_match.group(1).strip()
            reason = f"{analysis}\n\nFinal selection: {reason}"
        
        # Method 2: Try common patterns if XML didn't work
        if not selected:
            # Pattern: "Image 1" or "Image 2" or "选择图片1"
            img_match = re.search(r'[Ii]mage\s*(\d)|图片\s*(\d)|选择.*?(\d)', response_text)
            if img_match:
                selected = img_match.group(1) or img_match.group(2) or img_match.group(3)
        
        if not selected:
            # Pattern: "Answer: 1" or "答案：1" or "选择：1"
            ans_match = re.search(r'[Aa]nswer[:\s]+(\d)|答案[：:\s]+(\d)|选择[：:\s]+(\d)', response_text)
            if ans_match:
                selected = ans_match.group(1) or ans_match.group(2) or ans_match.group(3)
        
        if not selected:
            # Pattern: Look for "1" or "2" at the end or standalone
            end_match = re.search(r'\b([12])\s*[.。]?\s*$', response_text.strip())
            if end_match:
                selected = end_match.group(1)
        
        if not selected:
            # Pattern: Any standalone 1 or 2 in the response
            all_digits = re.findall(r'\b([12])\b', response_text)
            if all_digits:
                # Take the last occurrence (usually the final answer)
                selected = all_digits[-1]
        
        # Default to "1" if no selection found
        if not selected or selected not in ["1", "2"]:
            logging.warning(f"Could not parse selection from response, defaulting to 1")
            selected = "1"
        
        # Extract reason if not found via XML
        if not reason:
            # Try to extract meaningful text as reason
            # Remove thinking tags if present
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]+>', '', clean_text)  # Remove other XML tags
            clean_text = clean_text.strip()
            
            if len(clean_text) > 20:
                # Use the response text as reason (truncate if too long)
                reason = clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
            else:
                reason = f"Selected Image {selected}"
        
        # Include thinking for Gemini thinking models
        if thinking_text:
            reason = f"[Thinking]\n{thinking_text[:1000]}...\n\n[Analysis]\n{reason}"
        
        # Validate selection
        if selected not in ["1", "2"]:
            # Try to find any digit
            digit_match = re.search(r'\b([12])\b', response_text)
            selected = digit_match.group(1) if digit_match else "1"
        
        return selected, reason, response_text

    def judge_two_images(
        self, 
        image1: Image.Image, 
        image2: Image.Image, 
        user_query: str
    ) -> Tuple[str, str, str]:
        """
        Judge which image is better
        
        Returns:
            (selected_image, reason, response_text)
        """
        try:
            if self.api_type == "openai":
                return self._call_openai_api(image1, image2, user_query)
            elif self.api_type == "gemini":
                return self._call_gemini_api(image1, image2, user_query, use_thinking=False)
            elif self.api_type == "gemini_thinking":
                return self._call_gemini_api(image1, image2, user_query, use_thinking=True)
            else:
                raise ValueError(f"Unknown api_type: {self.api_type}")
        except Exception as e:
            logging.error(f"Judge API call failed: {e}")
            return "1", f"Error: {str(e)}", str(e)

    def compute(self, input_dict, model_dict):
        """Main compute method (same interface as LocalJudgeTwoImages)"""
        assert len(input_dict) == 1
        input_dict = input_dict[0]
        
        image1 = input_dict["image1"]
        image2 = input_dict["image2"]
        bbox1 = input_dict.get("bbox1")
        bbox2 = input_dict.get("bbox2")
        user_query = input_dict["user_query"]
        
        logging.info(f"APIJudgeTwoImages ({self.api_type}) - Starting judgment")
        logging.info(f"User query: {user_query}")
        logging.info(f"Box 1: {bbox1}")
        logging.info(f"Box 2: {bbox2}")
        
        selected_image, reason, response_text = self.judge_two_images(
            image1, image2, user_query
        )
        
        logging.info(f"API judgment result: Selected image {selected_image}")
        logging.info(f"Reason preview: {reason[:200]}...")
        
        # Build output
        output_dict = deepcopy(input_dict)
        output_dict["selected_image"] = selected_image
        output_dict["selected_box"] = selected_image  # Alias for compatibility
        output_dict["judge_reason"] = reason
        output_dict["judge_response"] = response_text
        output_dict["judge_method"] = f"api_{self.api_type}_{self.model}"
        
        # Set final bbox and point
        if selected_image == "1" and bbox1:
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            output_dict["image"] = image1
        elif selected_image == "2" and bbox2:
            output_dict["bbox_abs"] = bbox2
            output_dict["point_abs"] = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            output_dict["image"] = image2
        else:
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2) if bbox1 else None
            output_dict["image"] = image1
        
        return output_dict


# Convenience aliases
class OpenAIJudgeTwoImages(APIJudgeTwoImages):
    """OpenAI-compatible API judge (GPT-4o, Claude via OpenRouter, etc.)"""
    def __init__(self, api_key=None, base_url=None, model="gpt-4o", **kwargs):
        super().__init__(api_type="openai", api_key=api_key, base_url=base_url, model=model, **kwargs)


class GeminiJudgeTwoImages(APIJudgeTwoImages):
    """Google Gemini standard judge"""
    def __init__(self, api_key=None, model="gemini-2.0-flash", **kwargs):
        super().__init__(api_type="gemini", api_key=api_key, model=model, **kwargs)


class GeminiThinkingJudgeTwoImages(APIJudgeTwoImages):
    """Google Gemini Thinking Model judge (e.g., Gemini 2.5 Pro)"""
    def __init__(self, api_key=None, model="gemini-2.5-pro-preview-05-06", thinking_budget=8192, **kwargs):
        super().__init__(
            api_type="gemini_thinking", 
            api_key=api_key, 
            model=model, 
            thinking_budget=thinking_budget, 
            **kwargs
        )
