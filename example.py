from singlora import apply_singlora_to_model
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import copy
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup

# -----------------------------------------------------------------------------
# 3. Demonstration with a Transformer Model
# -----------------------------------------------------------------------------


def main():
    """
    Main function to demonstrate SingLoRA on a pre-trained transformer.
    """
    # --- Configuration ---
    MODEL_NAME = "distilbert-base-uncased"
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3  # As suggested for SingLoRA in the paper
    EPOCHS = 3
    RANK = 8
    ALPHA = 8.0
    RAMP_UP_STEPS = 1000  # T parameter from the paper
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {DEVICE}")

    # --- Load Model and Tokenizer ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Apply SingLoRA ---
    print("\nApplying SingLoRA to the model...")
    # We create a deepcopy to compare the original vs modified model structure
    original_model = copy.deepcopy(model)
    apply_singlora_to_model(
        model,
        rank=RANK,
        alpha=ALPHA,
        ramp_up_steps=RAMP_UP_STEPS,
        target_modules=["q_lin", "k_lin", "v_lin"],
    )
    model.to(DEVICE)

    print("\n--- Original Model Structure (Sample) ---")
    print(
        original_model.distilbert.transformer.layer[0].attention.q_lin
    )

    print(
        "\n--- Model Structure After Applying SingLoRA (Sample) ---"
    )
    print(model.distilbert.transformer.layer[0].attention.q_lin)

    # --- Count Trainable Parameters ---
    def count_trainable_parameters(m):
        return sum(
            p.numel() for p in m.parameters() if p.requires_grad
        )

    original_params = count_trainable_parameters(original_model)
    singlora_params = count_trainable_parameters(model)

    print(f"\nOriginal trainable parameters: {original_params:,}")
    print(f"SingLoRA trainable parameters: {singlora_params:,}")
    print(
        f"Parameter reduction: {100 * (1 - singlora_params / original_params):.2f}% (compared to full fine-tuning)"
    )

    # --- Prepare Data (Dummy Example) ---
    # In a real scenario, you would load a dataset like MNLI from the GLUE benchmark.
    print("\nCreating a dummy dataset for demonstration...")
    texts = [
        "SingLoRA is a novel method for fine-tuning.",
        "I love PyTorch!",
    ] * 20
    labels = [1, 0] * 20

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # --- Training Setup ---
    # Note: We only pass the parameters of SingLoRA layers to the optimizer.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # --- Training Loop ---
    print("\nStarting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = [t.to(DEVICE) for t in batch]
            b_input_ids, b_attention_mask, b_labels = batch

            optimizer.zero_grad()

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.4f}"
        )

    print("\nTraining finished successfully!")
    print(
        "Note the 'training_step' counter in a SingLoRA layer has been updated:"
    )
    print(
        f"Final training step for one layer: {model.distilbert.transformer.layer[0].attention.q_lin.training_step.item():.0f}"
    )


if __name__ == "__main__":
    # To avoid issues with multiprocessing in some environments

    main()
