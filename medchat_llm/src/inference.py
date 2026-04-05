from unsloth import FastLanguageModel
import torch 

model,tokenizer = FastLanguageModel.from_pretrained(
    "../models/medchat",
    load_in_4bit=True,  # load the model in 4-bit precision (saves VRAM and speeds up inference)
)

FastLanguageModel.for_inference(model)

def generate(prompt):
    inputs = tokenizer(prompt,return_tensors="pt").to("cuda")


    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature = 0.7,
    )
    return tokenizer.decode(outputs[0],skip_special_tokens=True)


if __name__ == "__main__":
    prompt = """<|system|>
You are a medical assistant.

<|user|>
What are symptoms of diabetes?

<|assistant|>
"""
    print(generate(prompt))