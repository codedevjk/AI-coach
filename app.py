# app.py
import gradio as gr
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import librosa
import time
import os
# --- Configuration ---
# Whisper model name should be the local name (e.g., 'tiny', 'base')
# 'tiny' is multilingual, 'tiny.en' is English-only but faster/ more accurate for English
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "google/gemma-2b-it") # Hugging Face model ID
HF_TOKEN = os.getenv("HF_TOKEN") # Get token from environment variable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Models (on startup) ---
print("Loading Whisper model...")
# Whisper expects the local model name directly
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

print("Loading LLM tokenizer and model...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=HF_TOKEN)
# Load model with optimizations for CPU/Memory if needed
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto", # Automatically distribute across available devices
    # low_cpu_mem_usage=True # Can help on limited RAM, might be implicit with device_map
)

# --- Helper Functions ---

def get_interview_questions(role):
    """Provides a list of questions based on the selected job role."""
    questions = {
        "ml_engineer": [
            "Describe a machine learning project you've worked on.",
            "How do you handle overfitting in your models?",
            "Explain the bias-variance tradeoff."
        ],
        "data_analyst": [
            "How do you approach cleaning a messy dataset?",
            "Tell me about a time you discovered an interesting insight in data.",
            "What tools do you use for data visualization?"
        ],
        "software_engineer": [
            "Explain the difference between a stack and a queue.",
            "Describe a challenging bug you fixed.",
            "How do you ensure your code is maintainable?"
        ]
    }
    return questions.get(role.lower(), ["Tell me about yourself.", "Why do you want this job?", "What are your strengths and weaknesses?"])

def transcribe_audio(audio_file_path):
    """Transcribes audio using Whisper."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return ""
    print(f"Transcribing audio: {audio_file_path}")
    # Whisper expects the file path directly
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    print(f"Transcription: {transcription}")
    return transcription

def analyze_audio_features(audio_file_path):
    """Analyzes audio for pace, pauses, etc."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return {"pace": "N/A", "filler_words": 0}
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None)
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        # Simple word count estimate (assuming average speaking rate)
        # A more robust method would involve ASR word timestamps
        words = transcribe_audio(audio_file_path).split()
        word_count = len(words)
        pace = word_count / duration if duration > 0 else 0 # words per second

        # Count filler words (basic example)
        filler_words = ["um", "uh", "like", "you know", "so", "basically"]
        filler_count = sum(1 for word in words if word.lower() in filler_words)

        return {"pace": f"{pace:.2f} words/sec", "filler_words": filler_count}
    except Exception as e:
        print(f"Error analyzing audio features: {e}")
        return {"pace": "Error", "filler_words": "Error"}

def generate_feedback(question, answer, role):
    """Generates feedback using the LLM."""
    if not answer.strip():
        return "Please provide an answer to receive feedback."

    # Define the prompt for the LLM
    prompt = f"""
You are an experienced career coach providing constructive feedback for a job interview practice session.

Job Role: {role}
Interview Question: {question}
Candidate's Answer: {answer}

Please provide feedback structured as follows:
1.  **Content Analysis:** Evaluate the relevance, depth, and accuracy of the answer in relation to the question and job role.
2.  **Structure (STAR):** Assess if the answer follows a clear structure (Situation, Task, Action, Result) where applicable.
3.  **Clarity:** Comment on how clear and concise the answer was.
4.  **Suggestions for Improvement:** Offer 1-2 specific suggestions to make the answer stronger.

Keep the feedback professional, encouraging, and actionable. Do not mention this prompt or your role as an AI.
"""
    print("Generating feedback with LLM...")
    inputs = llm_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        # Adjust max_new_tokens as needed
        outputs = llm_model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True, pad_token_id=llm_tokenizer.eos_token_id)
    feedback = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated Feedback: {feedback}")
    return feedback


# --- Gradio Interface ---
def process_interview(role, question, audio_file):
    """Main processing function that ties everything together."""
    start_time = time.time()
    transcription = transcribe_audio(audio_file)
    audio_features = analyze_audio_features(audio_file)
    feedback = generate_feedback(question, transcription, role)
    processing_time = time.time() - start_time
    return transcription, audio_features.get("pace", "N/A"), audio_features.get("filler_words", "N/A"), feedback, f"Processing completed in {processing_time:.2f} seconds."

# Define roles for dropdown
roles = ["ml_engineer", "data_analyst", "software_engineer", "general"]

with gr.Blocks(title="AI Interview Simulator") as demo:
    gr.Markdown("## 🎙️ AI Interview Simulator")
    gr.Markdown("Practice your interview skills and get instant AI feedback!")
    with gr.Row():
        role_dropdown = gr.Dropdown(choices=roles, label="Select Job Role", value="general")
        question_dropdown = gr.Dropdown(choices=[], label="Select Question")

    # Update questions based on role
    def update_questions(selected_role):
        questions = get_interview_questions(selected_role)
        return gr.update(choices=questions, value=questions[0] if questions else "")

    role_dropdown.change(fn=update_questions, inputs=role_dropdown, outputs=question_dropdown)

    audio_input = gr.Audio(label="Record Your Answer", type="filepath")
    transcribe_btn = gr.Button("Submit & Get Feedback")

    with gr.Column():
        transcription_output = gr.Textbox(label="Transcription")
        pace_output = gr.Textbox(label="Speaking Pace")
        filler_output = gr.Textbox(label="Filler Words Detected")
        feedback_output = gr.Textbox(label="AI Feedback", lines=10)
        status_output = gr.Textbox(label="Status")

    transcribe_btn.click(
        fn=process_interview,
        inputs=[role_dropdown, question_dropdown, audio_input],
        outputs=[transcription_output, pace_output, filler_output, feedback_output, status_output]
    )

    # Initialize questions on load
    demo.load(fn=update_questions, inputs=role_dropdown, outputs=question_dropdown)

# Launch the app, listening on all interfaces (important for Docker)
# The port must match EXPOSE in Dockerfile and app_port in README.md
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
