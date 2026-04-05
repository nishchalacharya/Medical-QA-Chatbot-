# Converts into chat format


def format_chat(example):
    return{
        "text":f"""<|system|>
    You are a helpful and safe medical assistant.
    You do not provide diagnosis.
    Always recommend consulting a doctor.

        <|user|>
        {example['question']}

        <|assistant|>
        {example['answer']}"""
        
    










    }