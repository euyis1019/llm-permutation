from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = Path(__file__).resolve().parents[1] / "models" / "Qwen3-4B"
prompt = "请只回答一个数字：9 的平方是多少？"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
)

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

answer = tokenizer.decode(
    output[0, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(f"Prompt: {prompt}")
print(f"Answer: {answer.strip()}")
