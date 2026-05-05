
import torch
import gradio as gr
from datetime import datetime
import json
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

SYSTEM_PROMPT = """You are SilentWitness, a legal rights assistant for India.
You help people understand their legal rights in simple language.
Always explain:
1. What law applies
2. What right the person has
3. What they should do next
Keep answers clear, in English and simple Hindi where helpful."""

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

def silentwitness(text_question, image_input, language_choice):
    if not text_question.strip():
        return "Please describe your legal problem.", ""
    lang_note = "Always respond in both English and simple Hindi." if language_choice == "Hindi + English" else "Respond in English only."
    if image_input is not None:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": SYSTEM_PROMPT + "\n" + lang_note + "\n\nAnalyze this document for legal issues. User says: " + text_question}
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT + "\n" + lang_note + "\n\nQuestion: " + text_question}
        ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", tokenize=True, return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7, do_sample=True)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("model\n")[-1].strip()
    evidence = {
        "case_id": f"SW-{datetime.now().strftime(\'%Y%m%d-%H%M%S\')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": text_question,
        "document_attached": image_input is not None,
        "legal_analysis": answer,
        "powered_by": "SilentWitness India — Gemma 4"
    }
    return answer, json.dumps(evidence, indent=2, ensure_ascii=False)
