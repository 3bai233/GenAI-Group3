import os
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from peft import PeftModel

def merge_checkpoint():
    # Path configuration
    base_model_path = "YOUR_BASE_MODEL_PATH"
    adapter_path = "YOUR_ADAPTER_PATH"

    # Auto-generate merged folder at the same level as original checkpoint
    output_path = f"{adapter_path}-merged"

    print(f"1. Loading base model: {base_model_path}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    print(f"2. Loading LoRA Checkpoint: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("3. Merging weights (Merge and Unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"4. Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    print("5. Saving Tokenizer and Processor (copied from base model)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("\n[OK] Merge completed!")
    print(f"You can now use {output_path} as a standalone model for inference.")

if __name__ == "__main__":
    merge_checkpoint()
