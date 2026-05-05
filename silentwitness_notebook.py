# SilentWitness India — Full Code
# Gemma 4 Good Hackathon by Google DeepMind
# Track: Digital Equity & Inclusivity + Safety & Trust

# ===== BLOCK 1: Install =====
# !pip install -q -U transformers accelerate gradio

# ===== BLOCK 2: Load Model =====
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Model loaded!")

# ===== BLOCK 3: Core Legal AI =====
SYSTEM_PROMPT = """You are SilentWitness, a legal rights assistant for India.
You help people understand their legal rights in simple language.
Always explain:
1. What law applies
2. What right the person has
3. What they should do next
Keep answers clear, in English and simple Hindi where helpful."""

def ask_legal_question(question):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": SYSTEM_PROMPT + "\n\nQuestion: " + question}
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", tokenize=True, return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)
    return processor.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()

# ===== BLOCK 4: Full Gradio App =====
import gradio as gr
from datetime import datetime
import json

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
    answer = processor.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()
    evidence = {
        "case_id": f"SW-{datetime.now().strftime(\'%Y%m%d-%H%M%S\')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": text_question,
        "document_attached": image_input is not None,
        "legal_analysis": answer,
        "powered_by": "SilentWitness India — Gemma 4"
    }
    return answer, json.dumps(evidence, indent=2, ensure_ascii=False)

with gr.Blocks(title="SilentWitness India", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ SilentWitness India\n### Free Legal Rights AI | Powered by Gemma 4")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload Document (optional)", type="pil", height=200)
            question = gr.Textbox(label="Describe your legal problem", lines=4,
                placeholder="Example: My landlord won\'t return my deposit...")
            language = gr.Radio(["Hindi + English", "English Only"], value="Hindi + English", label="Language")
            btn = gr.Button("⚖️ Know My Rights", variant="primary", size="lg")
        with gr.Column():
            answer_box = gr.Textbox(label="Legal Analysis", lines=16)
            evidence_box = gr.Textbox(label="Evidence Package", lines=8)
    gr.Examples(
        examples=[
            [None, "My landlord won\'t return my deposit after 3 months.", "Hindi + English"],
            [None, "My employer hasn\'t paid salary for 2 months.", "Hindi + English"],
            [None, "Police arrested me without a warrant. Is this legal?", "Hindi + English"],
            [None, "I bought a phone that broke after 1 week. Shop refuses refund.", "English Only"],
        ],
        inputs=[image, question, language]
    )
    btn.click(fn=silentwitness, inputs=[image, question, language], outputs=[answer_box, evidence_box])

demo.launch(share=True)
