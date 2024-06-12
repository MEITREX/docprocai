import peft
import transformers
import math
import datasets

class LlamaTrainer:
    lora_rank: int = 512
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 200
    training_epochs: int = 3
    learning_rate: float = 3e-4

    output_path: str = "output"

    model_id: str = "meta-llama/Meta-Llama-3-8B"

    def train(self, training_dataset: datasets.Dataset, eval_dataset: datasets.Dataset = None):
        bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config)

        lora_modules = peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["llama"]

        lora_config = peft.LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        lora_model = peft.get_peft_model(base_model, lora_config)

        class TrainingCallbacks(transformers.TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss']:.4f}")

        trainer = transformers.Trainer(
            model=lora_model,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.micro_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=self.warmup_steps,
                num_train_epochs=self.training_epochs,
                learning_rate=self.learning_rate,
                fp16=True,
                optim="adamw",
                logging_steps = 1,
                evaluation_strategy="no", # can use "steps" if we pass some eval dataset
                eval_steps=10,
                save_strategy="no",
                output_dir=self.output_path,
                use_ipex=True if transformers.is_torch_xpu_available() else False
            ),
            callbacks=list(TrainingCallbacks())
        )

        trainer.train()

        lora_model.save_pretrained(self.output_path)