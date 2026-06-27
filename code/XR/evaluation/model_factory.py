import os
import sys


def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "kimivl":
        from models.kimivl import KimiVLModel

        model = KimiVLModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel

        model = SeeClickModel()
        model.load_model()
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel

        model = Qwen1VLModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel

        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen2_5vl":
        from models.qwen2_5vl import Qwen2_5VLModel

        model = Qwen2_5VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif model_type == "qwen3vl":
        from models.qwen3vl import Qwen3VLModel

        model = Qwen3VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen3vl-thinking":
        from models.qwen3vl import Qwen3VLModel

        model = Qwen3VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen3vl_sft":
        from models.qwen3vl_sft import Qwen3VLSFTModel

        model = Qwen3VLSFTModel()
        model.load_model(
            model_name_or_path=args.model_name_or_path,
            max_pixels=args.max_pixels,
        )
        return model
    elif model_type == "holo1_5":
        from models.holo1_5 import Holo1_5Model

        model = Holo1_5Model()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel

        model = MiniCPMVModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel

        model = InternVLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel

        model = GPT4XModel()

    elif model_type == "gpt5":
        from models.gpt5 import GPT5Model

        model = GPT5Model()

    elif model_type == "gemini3":
        from models.gpt5 import GeminiModel

        model = GeminiModel()

    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel

        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel

        model = OSAtlas7BModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel

        model = UGroundModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif model_type == "fuyu":
        from models.fuyu import FuyuModel

        model = FuyuModel()
        model.load_model()
    elif model_type == "showui":
        from models.showui import ShowUIModel

        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel

        model = AriaUIVLLMModel()
        model.load_model()
    elif model_type == "cogagent":
        from models.cogagent import CogAgentModel

        model = CogAgentModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model

        model = CogAgent24Model()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif args.model_type == "opencua":
        from models.opencua import OpenCUAModel
        model = OpenCUAModel()
        model.load_model(args.model_name_or_path or "OpenCUA/OpenCUA-7B")
        return model

    # Methods
    elif model_type == "screenseeker":
        from models.methods.screenseeker import ScreenSeekeRMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = ScreenSeekeRMethod(planner="gpt-4o-2024-05-13", grounder=grounder)
    elif model_type == "reground":
        from models.methods.reground import ReGroundMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = ReGroundMethod(grounder=grounder)
    elif model_type == "iterative_narrowing":
        from models.methods.iterative_narrowing import IterativeNarrowingMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = IterativeNarrowingMethod(grounder=grounder)
    elif model_type == "iterative_focusing":
        from models.methods.iterative_focusing import IterativeFocusingMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = IterativeFocusingMethod(grounder=grounder)
    elif model_type == "mai_ui":
        # mai_ui_root = os.path.join(os.path.dirname(__file__), "..", "MAI-UI", "evaluation", "grounding")
        # mai_ui_root = os.path.abspath(mai_ui_root)
        # if mai_ui_root not in sys.path:
        #     sys.path.insert(0, mai_ui_root)

        from models.MAI_UI_V2 import CustomQwen3_VL_VLLM_Model

        model = CustomQwen3_VL_VLLM_Model()
        model.load_model(model_name_or_path=model_name_or_path, max_pixels=args.max_pixels)
    elif model_type == "tianxi_7b":
        from models.tianxi_7b import TianXi7BModel

        model = TianXi7BModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path, max_pixels=args.max_pixels)
        else:
            raise ValueError("tianxi7b requires --model_name_or_path")
    elif model_type == "ui_venus1_5":
        from models.ui_venus1_5 import UIVenus15Model

        model = UIVenus15Model()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            raise ValueError("ui_venus1_5 requires --model_name_or_path")
    elif model_type == "uitars":
        from models.uitars import UITarsModel

        model = UITarsModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            raise ValueError("uitars requires --model_name_or_path")
    elif args.model_type == "holo2":
        from models.holo2 import Holo2Model
        model = Holo2Model()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            raise ValueError("uitars requires --model_name_or_path")
    elif model_type == "gta1":
        from models.gta1 import GTA1Model

        model = GTA1Model()
        if model_name_or_path:
            model.load_model(
                model_name_or_path=model_name_or_path,
                max_pixels=args.max_pixels,
            )
        else:
            raise ValueError("gta1 requires --model_name_or_path")
    elif model_type == "seed1_5vl":
        from models.seed1_5vl import Seed1_5VLModel
        model = Seed1_5VLModel()
        model.load_model()
    elif model_type == "seed_venus_xr":
        from models.seed_venus_xr import SeedVenusXRGroundingModel

        model = SeedVenusXRGroundingModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "agent_s":
        from models.agent_s_grounding_full import AgentSGroundingWrapperFull
        
        model = AgentSGroundingWrapperFull()
        model.load_model(
            generation_model=args.provider if hasattr(args, 'provider') else "openai",
            generation_model_name=args.model if hasattr(args, 'model') else "gpt-4o",
            generation_url=args.model_url if hasattr(args, 'model_url') else os.environ.get("OPENAI_BASE_URL", ""),
            generation_api_key=args.model_api_key if hasattr(args, 'model_api_key') and args.model_api_key else os.environ.get("OPENAI_API_KEY", ""),

            router_model=args.router_provider if hasattr(args, 'router_provider') else "",
            router_model_name=args.router_model if hasattr(args, 'router_model') else "",
            router_url=args.router_url if hasattr(args, 'router_url') else "",
            router_api_key=args.router_api_key if hasattr(args, 'router_api_key') and args.router_api_key else "",
            
            grounding_model=args.ground_provider if hasattr(args, 'ground_provider') else "vllm",
            grounding_model_name=args.ground_model if hasattr(args, 'ground_model') else "MAI-UI-8B",
            grounding_url=args.ground_url if hasattr(args, 'ground_url') else "",
            grounding_api_key=args.ground_api_key if hasattr(args, 'ground_api_key') and args.ground_api_key else os.environ.get("OPENAI_API_KEY", ""),
            
            grounding_width=args.grounding_width if hasattr(args, 'grounding_width') else 1000,
            grounding_height=args.grounding_height if hasattr(args, 'grounding_height') else 1000,
            max_pixels=args.max_pixels if hasattr(args, 'max_pixels') else 3840*2160,
        )
        return model

    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    # Qwen3VL thinking output can be long; 256 may truncate before <tool_call>.
    default_max_new_tokens = 2048 if model_type == "qwen3vl-thinking" else 256
    model.set_generation_config(temperature=0, max_new_tokens=default_max_new_tokens)
    return model
