import os
import gc
import json
import torch
import torch.nn.functional as F
import glob
import random
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types

from data_processor import get_truncated_prompts
from rl_judge_prompts import RL_JUDGE_PROMPT

# Configuration
SFT_CHECKPOINT_PATH = "/content/drive/MyDrive/Qwen-Action-Breadth-SFT/checkpoint-2000"
OUTPUT_DIR = "/content/drive/MyDrive/Qwen-Action-Breadth-RLCAI"
JUDGE_MODEL = "gemini-3.5-flash"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

HYPERPARAMS = {
    "lr": 1e-5,
    "kl_coef": 0.05,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "save_steps": 50
}

def setup_rl_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    base_model.config.use_cache = False

    # Merge SFT weights
    policy_model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT_PATH)
    policy_model = policy_model.merge_and_unload()

    # Create new LoRA for RL
    lora_config = LoraConfig.from_pretrained(SFT_CHECKPOINT_PATH)
    lora_config.inference_mode = False
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.train()
    
    return policy_model, tokenizer

def compute_gated_reward(category, raw_score):
    if raw_score <= -10:
        return -10.0
    if category == "A":
        return float(raw_score)
    elif category in ["B", "C"]:
        if raw_score >= 8.0:
            return float(raw_score)
        elif raw_score >= 5.0:
            return 2.0
    return -10.0

def train_rl_step(prompt_text, policy_model, tokenizer, optimizer):
    policy_model.train()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(policy_model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Rollout Phase
    with torch.no_grad():
        policy_model.eval()
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=HYPERPARAMS["max_new_tokens"],
            temperature=HYPERPARAMS["temperature"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        policy_model.train()

    full_ids = outputs[0]
    response_ids = full_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Call LLM Judge
    full_transcript = f"{prompt_text}{response_text}"
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[f"Evaluate this conversation:\n\n{full_transcript}"],
            config=types.GenerateContentConfig(
                system_instruction=RL_JUDGE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text.strip())
        category, raw_score = data.get("category", "C"), float(data.get("score", -10))
    except:
        category, raw_score = "C", -10.0

    reward = compute_gated_reward(category, raw_score)

    # Calculate KL Penalty & Loss
    logits = policy_model(full_ids.unsqueeze(0)).logits[0, prompt_len-1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = response_ids.unsqueeze(-1)
    selected_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)

    with torch.no_grad():
        with policy_model.disable_adapter():
            ref_logits = policy_model(full_ids.unsqueeze(0)).logits[0, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_selected_log_probs = ref_log_probs.gather(dim=-1, index=target_ids).squeeze(-1)

    kl_per_token = selected_log_probs - ref_selected_log_probs
    kl_penalty_scalar = kl_per_token.sum().item()
    adjusted_reward = reward - (HYPERPARAMS["kl_coef"] * kl_penalty_scalar)
    
    policy_loss = - (selected_log_probs * adjusted_reward).mean()

    # Optimize
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    loss_val = policy_loss.item()
    torch.cuda.empty_cache()
    gc.collect()
    
    return loss_val, reward, raw_score, category
