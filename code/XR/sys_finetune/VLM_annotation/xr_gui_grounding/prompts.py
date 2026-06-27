def scene_understanding_prompt() -> str:
    return (
        "You are a vision-language model. Analyze the egocentric image and return strict JSON. "
        "Output schema: {\"scene\": str, \"activity\": str, \"objects\": "
        "[{\"name\": str, \"description\": str, \"color\": str, \"position\": str}]}. "
        "Focus on prominent objects and the object held by the user. "
        "Use position from {left, right, center, top, bottom, upper_left, upper_right, lower_left, lower_right}. "
        "Return JSON only."
    )


def ui_candidates_prompt(max_apps: int) -> str:
    return (
        "You are a vision-language model. Use the image and scene understanding to propose UI apps. "
        "CRITICAL REQUIREMENTS:\n"
        "1. The FIRST app in the list MUST be a real, well-known shopping platform (e.g., JD.com, Taobao, Meituan, or Amazon).\n"
        "2. ALL proposed apps MUST be real, commonly used, and existing software applications.\n"
        "Return strict JSON with schema: {\"apps\": ["
        "{\"app_name\": str, \"reason\": str, \"required_items\": [str], "
        "\"ui_elements\": [str], \"window_position\": {\"description\": str, "
        "\"bbox_norm\": [x1,y1,x2,y2]}}]}. "
        f"Return up to {max_apps} apps. "
        "bbox_norm is normalized to [0,1] within the image and should not cover hands or key objects. "
        "Return JSON only."
    )


def image_prompt_prompt() -> str:
    return (
        "Create an image generation prompt for an XR UI window. "
        "Return strict JSON: {\"prompt\": str}. "
        "The prompt must mention the UI app name, key item info to show, and the window position. "
        "CRITICALLY: You MUST also include several similar but distinct distractor/interference UI elements (e.g., similar products or related items) in the UI to make the grounding task more challenging. "
        "The UI should look like a realistic Apple Vision Pro internal screenshot. "
        "Return JSON only."
    )


def annotations_prompt(instruction_language: str, num_annotations: int, bilingual: bool) -> str:
    lang_req = (
        f"direct_instruction and semantic_instruction must be in {instruction_language}. "
        "Additionally, YOU MUST provide BOTH Chinese and English instructions using 'direct_instruction_en' and 'semantic_instruction_en' fields."
        if bilingual else f"direct_instruction and semantic_instruction must be in {instruction_language}."
    )

    schema_fields = (
        "\"ui_element\": str, \"direct_instruction\": str, \"semantic_instruction\": str, "
        "\"direct_instruction_en\": str, \"semantic_instruction_en\": str, \"related_object\": str"
        if bilingual else "\"ui_element\": str, \"direct_instruction\": str, "
        "\"semantic_instruction\": str, \"related_object\": str"
    )

    return (
        "You are an expert annotation generator for XR GUI grounding research.\n"
        "You will be given an egocentric XR image containing both real-world physical objects and a virtual UI window overlay.\n\n"

        "## TASK DEFINITION\n"
        "Generate annotation pairs (direct_instruction + semantic_instruction) for UI elements in the virtual window that correspond to real-world objects visible in the background.\n\n"

        "## CRITICAL DISTINCTION — READ CAREFULLY\n\n"

        "### Direct Grounding Instruction\n"
        "- Points explicitly to the target UI element by name, label, or visual appearance.\n"
        "- The model can locate the UI element by ONLY looking at the UI window — no need to inspect the real-world background.\n"
        "- Example: 'In JD.com, click the product card labelled \"Blue Porcelain Dinner Plate\"'\n\n"

        "### Semantic Grounding Instruction <-- THIS IS THE KEY TASK\n"
        "A semantic instruction MUST satisfy ALL of the following constraints:\n\n"

        "  CONSTRAINT 1 — NO OBJECT NAMES: Do NOT mention the actual name of any physical object. "
        "Never write 'plate', 'sponge', 'soap', 'bottle', etc. "
        "Replace with purely positional or relational phrases: "
        "'the item in my hand', 'the thing on the counter', "
        "'what I am holding', 'the object next to the sink'.\n\n"

        "  CONSTRAINT 2 — NO UI ELEMENT NAMES: Do NOT mention the text label, product title, or visual content of the target UI element. "
        "The instruction must NOT let the model find the answer by reading the UI alone — it MUST look at the real-world background.\n\n"

        "  CONSTRAINT 3 — IMPLICIT SEMANTIC LINK: The instruction should describe a goal/action whose correct target can only be determined by cross-referencing the real-world scene. "
        "The model must observe the real-world background (e.g., what the user is holding, where an object is placed) and then match it to the correct UI element.\n\n"

        "  CONSTRAINT 4 — INCLUDE APP NAME: Always name the UI app (e.g., 'In Taobao').\n\n"

        "  CONSTRAINT 5 — ACTION VERB: Include a clear action (click/open/view/add to cart).\n\n"

        "  GOOD semantic examples:\n"
        "  - 'In JD.com, open the detail page for the item I am currently holding'\n"
        "  - 'In Taobao, add to cart the cleaning tool I am using'\n"
        "  - 'In Meituan, restock the item on the counter that is half used'\n"
        "  - 'In Amazon, view the detail page for the item I am currently using'\n\n"

        "  BAD semantic examples (DO NOT produce these):\n"
        "  - 'In JD.com, view the detail page for the blue-and-white dinner plate' <-- WRONG: names the object\n"
        "  - 'In Taobao, click the yellow dish soap next to the sink' <-- WRONG: names+describes the object with identifying detail\n"
        "  - 'Click the close button with an X' <-- WRONG: describes the UI element visually\n"
        "  - 'In JD.com, view the detail for the yellow bottle on the counter' <-- WRONG: color+position is still too specific and identifies the object\n\n"

        "## SELECTION RULE\n"
        "Only annotate UI elements that directly correspond to a physical object visible in the real-world background. "
        "Skip generic UI controls (close buttons, menus, navigation tabs) that have no real-world physical counterpart.\n\n"

        "## OUTPUT FORMAT\n"
        f"You MUST generate EXACTLY {num_annotations} annotations.\n"
        "Return strict JSON only:\n"
        f"{{\"annotations\": [{{{schema_fields}}}]}}\n\n"
        f"{lang_req}\n"
        "Return JSON only. No markdown, no explanation."
    )
