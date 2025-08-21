# Meeting Summarizer Agent

An AI-powered application that records meetings, transcribes audio, detects language, translates if needed, and generates structured summaries using LLMs.

## Features

- Record audio directly from your microphone
- Transcribe audio using Whisper ASR
- Detect the language of the transcript
- Translate non-English transcripts to English
- Generate structured meeting summaries with:
  - Key Discussion Points
  - Action Items (with assigned owners if mentioned)
  - Key Decisions
- Export summaries as PDF
- Support for both OpenAI and Google Gemini LLMs
- Graphical User Interface (requires tkinter)

## Installation

### Cross-Platform Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Meeting_Summarizer_Agent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - For OpenAI: Get your API key from [OpenAI](https://platform.openai.com/api-keys)
   - For Google Gemini: Get your API key from [Google AI Studio](https://aistudio.google.com/)

### Windows-Specific Instructions

1. **Install PortAudio** (required for audio recording):
   - Download and install PortAudio from [http://www.portaudio.com/download.html](http://www.portaudio.com/download.html)
   - Or install using vcpkg:
     ```cmd
     vcpkg install portaudio
     ```

2. **Install PyAudio** (alternative audio library):
   ```cmd
   pip install pyaudio
   ```
   
   If you encounter issues with PyAudio, try:
   ```cmd
   pip install pipwin
   pipwin install pyaudio
   ```

3. **Install tkinter** (if not already installed with Python):
   - Usually included with standard Python installation on Windows
   - If missing, reinstall Python with the "tcl/tk and IDLE" option checked

### Linux-Specific Instructions

1. Install tkinter:
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On CentOS/RHEL: `sudo yum install tkinter` or `sudo dnf install python3-tkinter`

2. Install PortAudio development package:
   - On Ubuntu/Debian: `sudo apt-get install portaudio19-dev python3-pyaudio`
   - On CentOS/RHEL: `sudo yum install portaudio-devel`

### macOS-Specific Instructions

1. Install tkinter:
   - Usually included with Python installation
   - If missing: `brew install python-tk`

2. Install PortAudio:
   ```bash
   brew install portaudio
   ```

## Usage

### Console Mode (Original)

Run the application in console mode:
```bash
# Linux/macOS
python main.py

# Windows
python.exe main.py
```

This will process the `meeting_audio.mp3` file (or `.wav`) and generate a summary.

### UI Mode (New)

Run the application with the graphical user interface:
```bash
# Linux/macOS
python main.py --ui

# Windows
python.exe main.py --ui
```

In the UI, you can:
1. Select your LLM provider (OpenAI or Google Gemini)
2. Enter your API key
3. Record audio directly or select an existing audio file
4. Generate a structured summary of the meeting
5. Export the summary as a PDF

## How It Works

1. **Audio Recording**: The application can record audio directly from your microphone or process existing audio files.

2. **Transcription**: Uses faster-whisper for accurate speech-to-text conversion.

3. **Language Detection**: Automatically detects the language of the transcript using FastText.

4. **Translation**: If the detected language is not English, the transcript is translated using your chosen LLM.

5. **Summarization**: Generates a structured summary using either OpenAI's GPT-4o or Google's Gemini 1.5 Pro.

## File Structure

- `main.py`: Entry point for both console and UI modes
- `ui.py`: Graphical user interface implementation
- `agents/summary_agent.py`: Core logic for transcription, language detection, translation, and summarization
- `audio_export.py`: Audio recording functionality
- `requirements.txt`: List of required Python packages

## Troubleshooting

- If you encounter issues with audio recording, ensure your microphone permissions are granted
- For API-related errors, verify your API keys are correct and have sufficient credits
- If the application fails to start, ensure all dependencies are installed correctly
- If the UI doesn't start, make sure tkinter is installed
- On Windows, if audio recording fails, try installing PyAudio as an alternative

## Logging

The application generates log files for debugging:
- `meeting_summarizer.log`: Main application logs
- `audio_export.log`: Audio recording logs
- `summary_agent.log`: Summarization agent logs