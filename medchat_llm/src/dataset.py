def clean_example(example):
    question = example["question"].strip()
    answer = example["answer"].strip()

    # basic filtering 
    if len(answer) < 20:
        return None
    
    return {
        "question": question,
        "answer":answer
    }