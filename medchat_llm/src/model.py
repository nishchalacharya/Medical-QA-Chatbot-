# Loads model + applies LoRA

from unsloth import FastLanguageModel 

def load_model():
    model,tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tiny-llama-bnb-4bit",
        max_seq_length=1024,  # safer for my 8GB GPU (but can be increased to 2048 if you have more VRAM)
        load_in_4bit=True,  # load the model in 4-bit precision (saves VRAM and speeds up inference)
    )
    return model,tokenizer 


def apply_lora(model):
    # Apply LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # rank of the LoRA matrices (higher = more capacity, but also more VRAM usage)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"],  # which modules to apply LoRA to (these are the attention projection layers)  
        lora_alpha=32,  # scaling factor for the LoRA updates (higher = more aggressive fine-tuning, but can cause instability if too high
        lora_dropout = 0.05,  # dropout for the LoRA layers (helps prevent overfitting during fine-tuning
        use_gradient_checkpointing= True,  # use gradient checkpointing to save VRAM during training (trades off with slightly slower training speed

    )
    return model 