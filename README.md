# SingLoRA: A Minimal Implementation

This repository provides a minimal, single-file implementation of SingLoRA (Single Matrix Low-Rank Adaptation) as described in the paper ["SingLoRA: Low Rank Adaptation Using a Single Matrix"](https://arxiv.org/abs/2507.05566) by Bensaïd et al.

## Overview

SingLoRA is a parameter-efficient fine-tuning method that simplifies the LoRA architecture by using a single trainable matrix instead of two. This implementation demonstrates how to apply SingLoRA to transformer models using PyTorch and the Hugging Face Transformers library.

## Features

- Simple, self-contained implementation in a single Python file
- Compatible with Hugging Face Transformers models
- Includes a working example with DistilBERT
- Demonstrates parameter reduction compared to full fine-tuning

## Installation

```bash
pip3 install -U singlora
```

## Usage

### Basic Example

Here's a simple example of how to apply SingLoRA to a transformer model:

```python
from singlora import apply_singlora_to_model
from transformers import AutoModelForSequenceClassification

# Load your model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Apply SingLoRA
apply_singlora_to_model(
    model=model,
    rank=8,              # Low-rank dimension (r in the paper)
    alpha=8.0,           # Scaling factor
    ramp_up_steps=1000,  # Steps for ramp-up function u(t)
    target_modules=["q_lin", "k_lin", "v_lin"]  # Target attention layers
)

# Now only the SingLoRA parameters are trainable
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

### Configuration Parameters

- `rank`: The dimension of the low-rank adaptation (r). Lower values mean fewer parameters.
- `alpha`: Scaling factor for the adaptation. Higher values allow larger updates.
- `ramp_up_steps`: Number of steps (T) for the ramp-up function u(t) = min(t/T, 1).
- `target_modules`: List of layer names to apply SingLoRA to. Common targets:
  - `["query", "key", "value"]` for standard transformers
  - `["q_lin", "k_lin", "v_lin"]` for DistilBERT
  - `["q_proj", "k_proj", "v_proj"]` for LLaMA models

### Parameter Efficiency

SingLoRA significantly reduces the number of trainable parameters compared to full fine-tuning:

```python
# Example parameter counts
original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
singlora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

reduction = 100 * (1 - singlora_params / original_params)
print(f"Parameter reduction: {reduction:.2f}%")
```

## Complete Example

For a complete working example, see `example.py` in the repository.

### example.py output 
```txt
Applying SingLoRA to the model...
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.
Replaced 'q_lin' with SingLoRA layer.
Replaced 'k_lin' with SingLoRA layer.
Replaced 'v_lin' with SingLoRA layer.

--- Original Model Structure (Sample) ---
Linear(in_features=768, out_features=768, bias=True)

--- Model Structure After Applying SingLoRA (Sample) ---
SingLoRALayer(rank=8, alpha=8.0, ramp_up_steps=1000, original_layer=Linear(in_features=768, out_features=768, bias=True))

Original trainable parameters: 66,955,010
SingLoRA trainable parameters: 56,434,946
Parameter reduction: 15.71% (compared to full fine-tuning)

Creating a dummy dataset for demonstration...

Starting training...
Epoch 1/3 | Average Loss: 0.2366
Epoch 2/3 | Average Loss: 0.0002
Epoch 3/3 | Average Loss: 0.0000

Training finished successfully!
Note the 'training_step' counter in a SingLoRA layer has been updated:
Final training step for one layer: 15
```

### LLaMA Example

Here's how to apply SingLoRA to LLaMA models:

```python
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

# Example training setup
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Lower learning rate for LLaMA
)

# Example inference
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Key differences for LLaMA models:
- Use `LlamaForCausalLM` instead of standard transformer models
- Target the LLaMA-specific projection layers (`q_proj`, `k_proj`, `v_proj`)
- Consider using `float16` for memory efficiency
- Adjust hyperparameters (`rank`, `alpha`, learning rate) for larger models
- Use `device_map="auto"` for automatic model sharding on multiple GPUs

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@misc{bensaïd2025singloralowrankadaptation,
      title={SingLoRA: Low Rank Adaptation Using a Single Matrix}, 
      author={David Bensaïd and Noam Rotstein and Roy Velich and Daniel Bensaïd and Ron Kimmel},
      year={2025},
      eprint={2507.05566},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.05566}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
