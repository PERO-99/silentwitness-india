import torch
import gradio as gr
from datetime import datetime
import json
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

SYSTEM_PROMPT = """You are SilentWitness, a legal rights assistant for India.
Always explain: 1. What law applies 2. What right the person has 3. What to do next.
Keep answers clear, in English and simple Hindi."""

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

def silentwitness(text_question, language_choice):
    lang_note = "Respond in both English and simple Hindi." if language_choice == "Hindi + English" else "English only."
    messages = [{"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT + "\n" + lang_note + "\nQuestion: " + text_question}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7, do_sample=True)
    answer = processor.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()
    evidence = {"case_id": f"SW-{datetime.now().strftime('%Y%m%d-%H%M%S')}", "query": text_question, "legal_analysis": answer}
    return answer, json.dumps(evidence, indent=2)
