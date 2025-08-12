---
title: Ai Interview Simulator
emoji: 🏃
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# AI Interview Simulator

This repository contains an AI-powered Interview Simulator designed to help you practice for technical interviews. Select a job role, choose a common interview question, record your answer, and receive instant, constructive feedback from a Large Language Model (LLM).

## Features
*   **Role-Specific Questions**: Choose from over 40 technical roles, including React, Python, AWS, Docker, Machine Learning, and more.
*   **Audio Recording**: Use your microphone to record answers directly in the web interface.
*   **Accurate Transcription**: Your spoken answers are transcribed into text using OpenAI's Whisper model.
*   **AI-Generated Feedback**: Receive detailed feedback on your answer's content, clarity, and structure (STAR method), powered by Google's Gemma-2B-IT model.
*   **Simple Web Interface**: An easy-to-use interface built with Gradio.
*   **Dockerized**: Simple to set up and run locally using Docker, with support for both CPU and GPU.

## How it Works
The application follows a simple workflow:
1.  **Frontend (Gradio)**: The user selects a job role and a question from the corresponding list.
2.  **User Input**: The user records an audio response to the selected question.
3.  **Speech-to-Text (Whisper)**: The audio file is processed by the OpenAI Whisper model to generate an accurate text transcription.
4.  **Feedback Generation (Gemma)**: The question and the transcribed answer are sent to the Gemma-2B-IT model, which generates structured, actionable feedback based on a specialized prompt.
5.  **Display**: The transcription and the AI feedback are displayed to the user in the interface.

## How to Run Locally

### Prerequisites
*   [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.
*   A Hugging Face account and an Access Token. You can get a read-only token from your [Hugging Face settings page](https://huggingface.co/settings/tokens).

### Steps
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/codedevjk/ai-coach.git
    cd ai-coach
    ```

2.  **Create an environment file:**
    Create a file named `.env` in the root of the project and add your Hugging Face token. This is required to download the Gemma model.
    ```
    HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ```

3.  **Build the Docker image:**
    This command builds the image, installing Python, ffmpeg (required by Whisper), and all Python dependencies.
    ```sh
    docker build -t ai-coach .
    ```

4.  **Run the Docker container:**

    *   **For CPU:**
        The application is optimized to run on CPU with `low_cpu_mem_usage` settings.
        ```sh
        docker run -p 7860:7860 --rm -it --env-file .env ai-coach
        ```
    *   **For GPU (NVIDIA):**
        If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can run the container with GPU acceleration for significantly faster performance.
        ```sh
        docker run --gpus all -p 7860:7860 --rm -it --env-file .env ai-coach
        ```

5.  **Access the Application:**
    Once the container is running, open your web browser and navigate to `http://localhost:7860`.

## Configuration
You can configure the models used by the application via environment variables.

*   `HF_TOKEN`: (**Required**) Your Hugging Face access token.
*   `WHISPER_MODEL`: (Optional) The Whisper model to use for transcription. Defaults to `tiny`. Other options include `base`, `small`, `medium`, `large`. Larger models are more accurate but slower and require more resources.
*   `LLM_MODEL`: (Optional) The Hugging Face model ID for the feedback generator. Defaults to `google/gemma-2b-it`.

You can set these variables when running the `docker run` command. For example, to use the `base` Whisper model:
```sh
docker run -p 7860:7860 --rm -it --env-file .env -e WHISPER_MODEL=base ai-coach
```

## Technology Stack
*   **UI Framework**: Gradio
*   **Speech-to-Text**: OpenAI Whisper
*   **Language Model**: Google Gemma-2B-IT
*   **Backend**: Python, PyTorch, Hugging Face Transformers
*   **Containerization**: Docker
