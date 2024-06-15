import LlamaTrainer
import datasets
from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B"

def generate_training_prompt_tags():
    pass

def tokenize(prompt):
    

if __name__ == "__main__":
    trainer = LlamaTrainer.LlamaTrainer(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for prompt in generate_training_prompt_tags():
        tokens = tokenizer.encode(prompt)

    dataset = datasets.Dataset.from_list()
    trainer.train(dataset["train"])
