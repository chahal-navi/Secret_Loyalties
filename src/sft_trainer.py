import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def run_sft(raw_data_list, output_dir="/content/drive/MyDrive/Qwen-Action-Breadth-SFT"):
    model_path = snapshot_download(
        repo_id="Alamerton/12-mar-gen9-1.5b"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # OVERWRITE with the generation-aware ChatML template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}"
        "{% else %}"
        "{{ message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    # Memory Optimization & LoRA
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    dataset = Dataset.from_list(raw_data_list).shuffle(seed=42)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_steps=50,
        max_grad_norm=1.0,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting SFT training...")
    trainer.train()
