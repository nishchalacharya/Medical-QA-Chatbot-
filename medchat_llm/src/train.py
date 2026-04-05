import torch 
from transformers import TrainingArguments
from trl import SEFTTrainer 


from dataset import clean_dataset,load_medquad
from formatting import format_chat
from model import load_model,apply_lora


def main():
    print("Loading dataset...")
    dataset= load_medquad()
    dataset = clean_dataset(dataset)

    print("Formatting dataset...")
    dataset = dataset.map(format_chat)

    print("Loading model...")
    model,tokenizer = load_model()
    
    print("Applying LoRA...")
    model= apply_lora(model)

    print("Training...")

    trainer = SEFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 1024,
        packing = True,  # use packing to fit more examples into each batch (speeds up training and can improve performance, but can cause instability if your dataset has very long examples that exceed the max_seq_length when packed together
        args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir="../models",
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    trainer.train()

    print("Saving model ...")
    model.save_pretrained("../models/,medchat")


if __name__ == "__main__":
    main()










    



