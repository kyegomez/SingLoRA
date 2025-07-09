from singlora import apply_singlora_to_model
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # or your local path
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"           # Automatically handle model placement
)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Apply SingLoRA to attention layers
apply_singlora_to_model(
    model=model,
    rank=16,              # Can use larger rank for bigger models
    alpha=16.0,           # Increased alpha for stronger adaptation
    ramp_up_steps=2000,   # More steps for larger datasets
    target_modules=[      # LLaMA-specific attention layer names
        "q_proj",
        "k_proj",
        "v_proj"
    ]
)

print(model)