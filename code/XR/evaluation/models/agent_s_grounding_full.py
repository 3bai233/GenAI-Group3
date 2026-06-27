"""
Agent-S Framework Grounding Model Interface for ScreenSpot-Pro Evaluation (Full Version)

This module adapts the Agent-S framework's complete grounding capability for single-step
GUI element localization evaluation, including both the Generator (planning) agent and the
Grounding agent.

Pipeline:
1. Generator Agent (Planning Model): Analyzes the instruction and image to generate a detailed element description
2. Grounding Agent: Generates coordinates from the element description
"""

import os
import re
import sys
import io
import base64
import logging
import textwrap
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Agent-S to path
AGENT_S_PATH = os.path.join(os.path.dirname(__file__), "../../Agent-S")
if AGENT_S_PATH not in sys.path:
    sys.path.insert(0, AGENT_S_PATH)

from gui_agents.s3.agents.grounding import OSWorldACI, ACI
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from gui_agents.s3.utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)


class AgentSGroundingWrapperFull:
    """
    Full version wrapper for Agent-S grounding with planning + grounding pipeline.
    
    This wrapper implements the complete Agent-S flow:
    1. Generator Agent (Planning Model): Analyzes instruction and generates element description
    2. Grounding Agent: Generates coordinates from the element description
    """
    
    def __init__(self):
        """Initialize with empty attributes - actual initialization happens in load_model."""
        self.ground_provider = None
        self.ground_url = None
        self.ground_model = None
        self.ground_api_key = None
        self.grounding_width = None
        self.grounding_height = None
        self.platform = None
        self.max_trajectory_length = None
        
        self.engine_params_for_grounding = None
        self.engine_params_for_generation = None
        
        self.grounding_agent = None
        self.generator_agent = None
        self.task_router_agent = None
        self.spatial_hint_agent = None

        self.task_type_labels = {
            "direct": "Direct Grounding",
            "spatial": "Spatial Grounding",
            "semantic": "Semantic Grounding",
        }
        
    def load_model(self, 
                   generation_model: str = "openai",
                   generation_model_name: str = "gpt-4o",
                   generation_url: str = "",
                   generation_api_key: str = "",
                   router_model: str = "",
                   router_model_name: str = "",
                   router_url: str = "",
                   router_api_key: str = "",
                   grounding_model: str = "vllm",
                   grounding_model_name: str = "MAI-UI-8B",
                   grounding_url: str = "",
                   grounding_api_key: str = "",
                   grounding_width: int = 1000,
                   grounding_height: int = 1000,
                   platform: str = "linux",
                   max_trajectory_length: int = 8,
                   max_pixels: int = 3840 * 2160,
                   **kwargs):
        """
        Load the Agent-S grounding model with full pipeline (planning + grounding).
        
        Args:
            generation_model: Provider for the generator/planning model (e.g., "openai")
            generation_model_name: Generator model name (e.g., "gpt-4o")
            generation_url: URL for the generator model
            generation_api_key: API key for the generator model
            router_model: Provider for the task router model. If empty, reuse generation_model
            router_model_name: Task router model name. If empty, reuse generation_model_name
            router_url: URL for the task router model. If empty, reuse generation_url
            router_api_key: API key for the task router model. If empty, reuse generation_api_key
            grounding_model: Provider for the grounding model (e.g., "vllm")
            grounding_model_name: Grounding model name (e.g., "MAI-UI-8B")
            grounding_url: URL for the grounding model
            grounding_api_key: API key for the grounding model
            grounding_width: Width of the grounding model's coordinate output
            grounding_height: Height of the grounding model's coordinate output
            platform: OS platform
            max_trajectory_length: Maximum trajectory length for context management
            max_pixels: Not used, kept for compatibility
            **kwargs: Additional arguments for compatibility
        """
        def _resolve_api_key(engine_type: str, explicit_key: str) -> str:
            """Resolve api key with vLLM-friendly fallback for OpenAI-compatible servers."""
            if explicit_key:
                return explicit_key
            if (engine_type or "").lower() == "vllm":
                return (
                    os.environ.get("vLLM_API_KEY", "")
                    or os.environ.get("VLLM_API_KEY", "")
                    or "EMPTY"
                )
            return os.environ.get("OPENAI_API_KEY", "")

        # Store parameters
        resolved_grounding_api_key = _resolve_api_key(grounding_model, grounding_api_key)
        self.ground_provider = grounding_model
        self.ground_url = grounding_url
        self.ground_model = grounding_model_name
        self.ground_api_key = resolved_grounding_api_key
        self.grounding_width = grounding_width
        self.grounding_height = grounding_height
        self.platform = platform
        self.max_trajectory_length = max_trajectory_length
        
        # Engine parameters for grounding model
        self.engine_params_for_grounding = {
            "engine_type": grounding_model,
            "model": grounding_model_name,
            "base_url": grounding_url,
            "api_key": resolved_grounding_api_key,
            "grounding_width": grounding_width,
            "grounding_height": grounding_height,
        }

        # Router model defaults to generation model when not specified.
        router_engine_type = router_model or generation_model
        router_engine_name = router_model_name or generation_model_name
        router_engine_url = router_url if router_url else generation_url
        router_engine_api_key = _resolve_api_key(router_engine_type, router_api_key if router_api_key else generation_api_key)
        
        # Engine parameters for generator/planning model
        self.engine_params_for_generation = {
            "engine_type": generation_model,
            "model": generation_model_name,
            "base_url": generation_url,
            "api_key": _resolve_api_key(generation_model, generation_api_key),
        }

        self.engine_params_for_router = {
            "engine_type": router_engine_type,
            "model": router_engine_name,
            "base_url": router_engine_url,
            "api_key": router_engine_api_key,
        }
        
        # Initialize the grounding agent (OSWorldACI)
        self.grounding_agent = OSWorldACI(
            env=None,
            platform=platform,
            engine_params_for_generation=self.engine_params_for_generation,
            engine_params_for_grounding=self.engine_params_for_grounding,
            width=grounding_width,
            height=grounding_height,
        )
        
        # Initialize the generator agent (planning model)
        self._init_generator_agent()
        self._init_task_router_agent()
        self._init_spatial_hint_agent()

    def _call_agent_once_with_image(self, agent: LMMAgent, user_message: str, image: Image.Image) -> str:
        """Call a single-turn multimodal agent and reset history to system prompt afterwards."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        agent.add_message(user_message, image_content=image_bytes, role="user")
        response = call_llm_safe(agent)
        agent.add_message(response, role="assistant")

        if len(agent.messages) > 1:
            agent.messages = [agent.messages[0]]
        return response

    def _init_task_router_agent(self):
        """Initialize task router agent for Direct/Spatial/Semantic classification."""
        router_prompt = textwrap.dedent("""
            You are an XR GUI grounding task classifier.

            Given an instruction and XR screenshot, classify into exactly one category:
            1) Direct Grounding
            2) Spatial Grounding
            3) Semantic Grounding

            Definitions:
            - Direct Grounding:
              Instruction is simple and target can be identified from UI context alone.
              No modifier that refers to real-world XR background objects, and no relative spatial relation.
              Examples: "Change to takeout", "Add an order of McNuggets".
            - Spatial Grounding:
              Instruction relies on explicit relative position to a reference entity.
              Cues: above, below, left of, right of, next to, near, on the wall, 上方, 下方, 左侧, 右侧, 旁边, 墙上.
            - Semantic Grounding:
              Instruction refers to real-world object identity/meaning and needs semantic understanding from image.
              Cues: in my hand, in front of me, on the table, same item/brand, 我手中的, 我面前的, 桌上的, 同款.

            Decision order:
            1) If no spatial relation and no real-world-object semantic reference -> Direct Grounding.
            2) Else if explicit spatial relation exists -> Spatial Grounding.
            3) Else if real-world-object semantic reference exists -> Semantic Grounding.
            4) Else -> Direct Grounding.

            Output ONLY in this format:
            <task_type>Direct Grounding|Spatial Grounding|Semantic Grounding</task_type>
            <reason>[one concise sentence]</reason>
        """)

        router_engine_params = {
            "engine_type": self.engine_params_for_router["engine_type"],
            "model": self.engine_params_for_router["model"],
            "api_key": self.engine_params_for_router.get("api_key", ""),
            "base_url": self.engine_params_for_router.get("base_url", ""),
        }

        self.task_router_agent = LMMAgent(engine_params=router_engine_params)
        self.task_router_agent.add_message(router_prompt, role="system")

    def _init_spatial_hint_agent(self):
        """Initialize planning agent used to generate short spatial location hints."""
        spatial_prompt = textwrap.dedent("""
            You are a GUI spatial analyzer.
            Given an instruction and screenshot, output one short sentence describing
            where the target UI window/element is located in the image.

            Keep it short and concrete (for example: "in the upper-left area", "near the right-middle").

            Output ONLY in this format:
            <spatial_hint>[one short sentence]</spatial_hint>
        """)

        spatial_engine_params = {
            "engine_type": self.engine_params_for_generation["engine_type"],
            "model": self.engine_params_for_generation["model"],
            "api_key": self.engine_params_for_generation.get("api_key", ""),
            "base_url": self.engine_params_for_generation.get("base_url", ""),
        }

        self.spatial_hint_agent = LMMAgent(engine_params=spatial_engine_params)
        self.spatial_hint_agent.add_message(spatial_prompt, role="system")

    def _parse_task_type(self, response: str) -> str:
        """Parse task type from router response and map to direct|spatial|semantic."""
        match = re.search(r"<task_type>(.*?)</task_type>", response, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""

        text = match.group(1).strip()
        normalized = text.lower()

        if "direct" in normalized:
            return "direct"
        if "spatial" in normalized:
            return "spatial"
        if "semantic" in normalized:
            return "semantic"
        return ""

    def _parse_router_reason(self, response: str) -> str:
        """Extract optional reasoning text from router response."""
        match = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        for line in response.splitlines():
            text = line.strip()
            if text:
                return text[:200]
        return ""

    def _rule_based_task_type(self, instruction: str) -> str:
        """Deterministic fallback classifier when router output is unstructured."""
        normalized = (instruction or "").lower()

        spatial_cues = [
            "above", "below", "left of", "right of", "next to", "near", "behind",
            "on the wall", "上方", "下方", "左侧", "右侧", "旁边", "墙上",
        ]
        semantic_cues = [
            "in my hand", "in front of me", "on the table", "same item", "same brand",
            "我手中的", "我面前的", "桌上的", "同款",
        ]

        has_spatial = any(cue in normalized for cue in spatial_cues)
        has_semantic = any(cue in normalized for cue in semantic_cues)

        if not has_spatial and not has_semantic:
            return "direct"
        if has_spatial:
            return "spatial"
        if has_semantic:
            return "semantic"
        return "direct"

    def _classify_task_type(self, instruction: str, image: Image.Image) -> Tuple[str, str, str]:
        """Classify task into direct/spatial/semantic and return parsed + raw output."""
        user_message = f"""
Instruction:
\"{instruction}\"

Classify this grounding request using the screenshot.
Return ONLY:
<task_type>Direct Grounding|Spatial Grounding|Semantic Grounding</task_type>
<reason>[one concise sentence]</reason>
"""
        try:
            response = self._call_agent_once_with_image(self.task_router_agent, user_message, image)
            task_type = self._parse_task_type(response)
            reason = self._parse_router_reason(response)

            if not task_type:
                print("[STAGE 0 WARN] Router output is not in expected tag format.")
                print(f"[STAGE 0 WARN] Instruction: {instruction}")
                print("[STAGE 0 WARN] Expected tags: <task_type>...</task_type> and <reason>...</reason>")
                print("[STAGE 0 WARN] Router raw output:")
                print(response)
                task_type = self._rule_based_task_type(instruction)
                if reason:
                    reason = f"fallback_by_instruction_rules -> {task_type}; router_text: {reason}"
                else:
                    reason = f"fallback_by_instruction_rules -> {task_type}"

            return task_type, reason, response
        except Exception as e:
            logger.error(f"Error classifying task type: {e}")
            task_type = self._rule_based_task_type(instruction)
            return task_type, f"router_error_fallback -> {task_type}: {e}", ""

    def _parse_spatial_hint(self, response: str) -> str:
        """Extract <spatial_hint> content from model response."""
        match = re.search(r"<spatial_hint>(.*?)</spatial_hint>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        text = response.strip().split("\n")[0].strip()
        return text[:160] if text else "near the center area"

    def _contains_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _build_spatial_augmented_instruction(self, instruction: str, spatial_hint: str) -> str:
        """Append spatial hint to instruction with language-aware template."""
        if self._contains_chinese(instruction):
            return f"{instruction}。补充空间线索：目标UI窗口位于{spatial_hint}。"
        return f"{instruction} Additional spatial hint: the target UI window is located {spatial_hint}."

    def _generate_spatial_hint(self, instruction: str, image: Image.Image) -> Tuple[str, str]:
        """Generate a short spatial hint sentence from the planning model."""
        user_message = f"""
Instruction:
\"{instruction}\"

Analyze screenshot and provide one concise location hint.
"""
        try:
            response = self._call_agent_once_with_image(self.spatial_hint_agent, user_message, image)
            hint = self._parse_spatial_hint(response)
            return hint, response
        except Exception as e:
            logger.error(f"Error generating spatial hint: {e}")
            return "near the center area", ""

    def _ground_with_instruction(self, grounding_instruction: str, image: Image.Image) -> Dict[str, Any]:
        """Run grounding model once and return both raw and normalized coordinates."""
        image_bytes = self.image_to_bytes(image)
        obs = {
            "screenshot": image_bytes,
        }
        self.grounding_agent.assign_screenshot(obs)

        coords = self.grounding_agent.generate_coords(grounding_instruction, obs)
        raw_x, raw_y = coords[0], coords[1]

        norm_x = raw_x / self.grounding_width
        norm_y = raw_y / self.grounding_height
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        return {
            "raw_coords": [raw_x, raw_y],
            "norm_coords": [norm_x, norm_y],
        }
        
    def _init_generator_agent(self):
        """Initialize the generator agent with system prompt."""
        # Create system prompt for element description generation
        sys_prompt = textwrap.dedent("""
            You are a UI element analyzer. Your task is to analyze the user's instruction 
            and the current screenshot to generate a detailed description of the target UI element.
            
            Given the user's instruction and the screenshot, you should:
            1. Understand what UI element the user wants to interact with
            2. Generate a detailed, specific description of that element including:
               - The element type (button, link, icon, text field, etc.)
               - The text/label on the element (if any)
               - Visual characteristics (color, size, position hints)
               - Any distinguishing features
            
            Output your response in the following format:
            <thinking>
            [Your analysis of the instruction and what element to look for]
            </thinking>
            
            <element_description>
            [Detailed description of the target UI element]
            </element_description>
        """)
        
        # Create engine params for the generator agent
        generator_engine_params = {
            "engine_type": self.engine_params_for_generation["engine_type"],
            "model": self.engine_params_for_generation["model"],
            "api_key": self.engine_params_for_generation.get("api_key", ""),
            "base_url": self.engine_params_for_generation.get("base_url", ""),
        }
        
        # Create the generator agent
        self.generator_agent = LMMAgent(engine_params=generator_engine_params)
        
        # Set system prompt
        self.generator_agent.add_message(sys_prompt, role="system")
        
    def _generate_element_description(self, instruction: str, image: Image.Image) -> str:
        """
        Use the generator agent (planning model) to generate a detailed element description.
        
        Args:
            instruction: User's instruction
            image: PIL Image
            
        Returns:
            Detailed element description string
        """
        # Create user message with instruction
        user_message = f"""
Task: Locate and describe the following UI element:
"{instruction}"

Please analyze the screenshot and provide a detailed description of the target element.
"""
        
        print(f"[DEBUG - Planning Model] Sending request to generator agent...")
        print(f"[DEBUG - Planning Model] Instruction: {instruction}")
        
        # Call the planning model
        try:
            response = self._call_agent_once_with_image(self.generator_agent, user_message, image)
            print(f"[DEBUG - Planning Model] Raw response:\n{response}")
            
            # Extract element description from response
            element_description = self._parse_element_description(response)
            print(f"[DEBUG - Planning Model] Extracted element description: {element_description}")
            
            return element_description
            
        except Exception as e:
            logger.error(f"Error calling planning model: {e}")
            print(f"[DEBUG - Planning Model] Error: {e}")
            # Fallback: use original instruction
            return instruction
    
    def _parse_element_description(self, response: str) -> str:
        """
        Parse the element description from the generator agent's response.
        
        Args:
            response: Raw response from generator agent
            
        Returns:
            Extracted element description
        """
        # Try to extract from <element_description> tags
        match = re.search(r'<element_description>(.*?)</element_description>', response, re.DOTALL)
        if match:
            description = match.group(1).strip()
            return description
        
        # If no tags found, try to extract from common patterns
        # Look for text after common markers
        markers = [
            "Target element:",
            "Element to locate:",
            "UI element:",
            "The element",
            "Description:"
        ]
        
        for marker in markers:
            if marker in response:
                idx = response.find(marker)
                # Extract text after marker, up to a newline or end
                text = response[idx + len(marker):].strip()
                # Take first line or up to 200 chars
                end_idx = text.find('\n')
                if end_idx == -1:
                    end_idx = min(200, len(text))
                return text[:end_idx].strip()
        
        # If all else fails, return the original response cleaned up
        # Remove thinking tags if present
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        cleaned = cleaned.strip()
        # Limit length
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "..."
        return cleaned if cleaned else response
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")
    
    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert a PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def ground_only_positive(self, instruction: str, image: Any) -> Dict[str, Any]:
        """
        Perform grounding with full pipeline: planning model + grounding model.
        
        Pipeline:
        1. Planning Model (Generator Agent): Analyzes instruction and generates detailed element description
        2. Grounding Model: Generates coordinates from the element description
        
        Args:
            instruction: Text description of the UI element to locate
            image: Either a file path (str) or a PIL Image object
            
        Returns:
            Dictionary with keys:
                - "result": "positive" or "negative"
                - "format": "point"
                - "point": [x, y] normalized coordinates in [0, 1] range, or None
                - "bbox": None
                - "raw_response": The raw response containing both planning and grounding outputs
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        assert isinstance(image, Image.Image), "Invalid input image."
        
        # Get image dimensions
        img_width, img_height = image.size
        
        print(f"\n{'='*60}")
        print(f"[AGENT-S FULL PIPELINE] Starting grounding process")
        print(f"{'='*60}")
        print(f"[INPUT] Original instruction: {instruction}")
        print(f"[INPUT] Image size: {img_width}x{img_height}")
        
        try:
            # ============================================================
            # STAGE 0: Task Routing
            # ============================================================
            print(f"\n{'='*60}")
            print(f"[STAGE 0] Task Type Classification")
            print(f"{'='*60}")

            task_type, task_reason, router_raw = self._classify_task_type(instruction, image)
            predicted_task_type = self.task_type_labels.get(task_type, "Semantic Grounding")
            print(f"[STAGE 0 OUTPUT] Predicted task type: {predicted_task_type}")
            if task_reason:
                print(f"[STAGE 0 OUTPUT] Reason: {task_reason}")

            element_description = None
            spatial_hint = None
            spatial_raw = ""

            # ============================================================
            # STAGE 1: Route-specific planning
            # ============================================================
            print(f"\n{'='*60}")
            print(f"[STAGE 1] Route-specific planning")
            print(f"{'='*60}")

            if task_type == "direct":
                routed_instruction = instruction
                print("[STAGE 1 ROUTE] Direct Grounding -> skip Agent-S planning")
            elif task_type == "spatial":
                print("[STAGE 1 ROUTE] Spatial Grounding -> generate spatial hint and append to instruction")
                spatial_hint, spatial_raw = self._generate_spatial_hint(instruction, image)
                routed_instruction = self._build_spatial_augmented_instruction(instruction, spatial_hint)
                print(f"[STAGE 1 OUTPUT] Spatial hint: {spatial_hint}")
                print(f"[STAGE 1 OUTPUT] Routed instruction: {routed_instruction}")
            else:
                print("[STAGE 1 ROUTE] Semantic Grounding -> use Agent-S planning output")
                element_description = self._generate_element_description(instruction, image)
                routed_instruction = element_description
                print(f"[STAGE 1 OUTPUT] Element description: {element_description}")

            # ============================================================
            # STAGE 2: Grounding Model
            # ============================================================
            print(f"\n{'='*60}")
            print(f"[STAGE 2] Grounding Model")
            print(f"{'='*60}")
            print(f"[STAGE 2 INPUT] Grounding instruction: {routed_instruction}")

            grounded = self._ground_with_instruction(routed_instruction, image)
            raw_x, raw_y = grounded["raw_coords"]
            norm_x, norm_y = grounded["norm_coords"]

            print(f"[STAGE 2 OUTPUT] Raw coordinates: [{raw_x}, {raw_y}]")
            print(f"[STAGE 2 OUTPUT] Grounding model config: {self.grounding_width}x{self.grounding_height}")
            print(f"[STAGE 2 OUTPUT] Normalized coordinates: [{norm_x:.4f}, {norm_y:.4f}]")

            raw_response = f"""
[STAGE 0 - Task Routing]
Instruction: {instruction}
Predicted Task Type: {predicted_task_type}
Reason: {task_reason}
Router Raw Output: {router_raw}

[STAGE 1 - Route-specific Planning]
Spatial Hint: {spatial_hint}
Spatial Raw Output: {spatial_raw}
Element Description: {element_description}
Grounding Instruction: {routed_instruction}

[STAGE 2 - Grounding Model]
Raw Coordinates: [{raw_x}, {raw_y}]
Normalized Coordinates: [{norm_x:.4f}, {norm_y:.4f}]
Grounding Model Resolution: {self.grounding_width}x{self.grounding_height}
"""
            
            result = {
                "result": "positive",
                "format": "point",
                "point": [norm_x, norm_y],
                "bbox": None,
                "raw_response": raw_response,
                "predicted_task_type": predicted_task_type,
                "routing_strategy": task_type,
                "routing_reason": task_reason,
                "routed_instruction": routed_instruction,
            }
            
            print(f"\n{'='*60}")
            print(f"[AGENT-S FULL PIPELINE] Completed successfully")
            print(f"{'='*60}")
            print(f"[RESULT] Coordinates: [{norm_x:.4f}, {norm_y:.4f}]")
            
            return result
            
        except Exception as e:
            print(f"[STAGE 2 ERROR] Grounding failed: {e}")
            print(f"\n{'='*60}")
            print(f"[AGENT-S FULL PIPELINE] Failed")
            print(f"{'='*60}")
            
            # If grounding fails, return negative result
            return {
                "result": "negative",
                "format": "point",
                "point": None,
                "bbox": None,
                "raw_response": f"Error in Stage 2 (Grounding): {str(e)}",
                "predicted_task_type": "Semantic Grounding",
                "routing_strategy": "semantic",
                "routing_reason": "fallback_after_error",
                "routed_instruction": instruction,
            }
