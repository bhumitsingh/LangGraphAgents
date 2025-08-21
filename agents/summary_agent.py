import os
import requests
import logging
import sys
from typing import TypedDict, Annotated, List
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, END
from faster_whisper import WhisperModel
import fasttext
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("summary_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()

LLM_PROVIDER = "openai" # Change this to "gemini" for different llm provider
logger.debug(f"LLM provider set to: {LLM_PROVIDER}")

# Use forward slashes for compatibility, but normalize for the current OS
model_path = os.path.normpath("lid.176.bin")
logger.debug(f"Checking for language detection model at: {model_path}")
if not os.path.exists(model_path):
    logger.info(f"Downloading FastText language model to '{model_path}'...")
    print(f"Downloading FastText language model to '{model_path}'...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    r = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    logger.info("Language model download complete")
    print("Download complete.")

# Initialize Models
logger.info("Initializing Whisper transcription model")
transcription_model = WhisperModel("base", device="cpu", compute_type="int8")
logger.info("Loading FastText language detection model")
language_model = fasttext.load_model(model_path)

# Initialize LLM based on choice
logger.info(f"Initializing LLM with provider: {LLM_PROVIDER}")
if LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key not found for the chosen provider")
        raise ValueError("OpenAI API key not found for the chosen provider")
    logger.info("Using OpenAI (gpt-4o) as the LLM")
    print("Using OpenAI (gpt-4o) as the LLM")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
elif LLM_PROVIDER == "gemini":
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("Google API key not found for the chosen provider")
        raise ValueError("Google API key not found for the chosen provider")
    logger.info("Using Google (Gemini 1.5 Pro) as the LLM")
    print("Using Google (Gemini 1.5 Pro) as the LLM")
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro-latest",
        temperature=0,
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        },
    )
else:
    logger.error(f"Invalid LLM_PROVIDER: '{LLM_PROVIDER}'. Choose 'openai' or 'gemini'.")
    raise ValueError(f"Invalid LLM_PROVIDER: '{LLM_PROVIDER}'. Choose 'openai' or 'gemini'.")

# State for the Graph

class AgentState(TypedDict):
    audio_file_path: str
    original_transcript: str
    detected_language: str
    final_transcript: str
    summary: str

def transcribe_audio_node(state: AgentState) -> dict:
    """Transcribes the audio file to text using faster-whisper."""
    logger.info("--- 🔊 Transcribing Audio ---")
    print("--- 🔊 Transcribing Audio ---")
    audio_path = state.get("audio_file_path")
    # Normalize path for cross-platform compatibility
    if audio_path:
        audio_path = os.path.normpath(audio_path)
    logger.debug(f"Audio file path: {audio_path}")
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file not found at: {audio_path}")
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    logger.info("Starting audio transcription with Whisper model")
    segments, info = transcription_model.transcribe(audio_path, beam_size=5)
    logger.debug(f"Transcription info - language: {info.language}, duration: {info.duration}")
    transcript = " ".join([segment.text for segment in segments])
    logger.debug(f"Transcript length: {len(transcript)} characters")
    print("Transcription complete.")
    logger.info("Audio transcription completed")
    return {"original_transcript": transcript}

def detect_language_node(state: AgentState) -> dict:
    """Detects the primary language of the transcript."""
    logger.info("--- 🌐 Detecting Language ---")
    print("--- 🌐 Detecting Language ---")
    transcript = state.get("original_transcript")
    logger.debug(f"Original transcript length: {len(transcript) if transcript else 0} characters")
    if not transcript:
        logger.warning("No transcript provided for language detection")
        print("No transcript available for language detection.")
        return {"detected_language": ""}
    cleaned_transcript = transcript.replace("\n", " ")
    logger.debug(f"Cleaned transcript length: {len(cleaned_transcript)} characters")
    logger.info("Running language detection with FastText model")
    predictions = language_model.predict(cleaned_transcript, k=1)
    lang_code = predictions[0][0].replace("__label__", "") if predictions[0] else ""
    confidence = predictions[1][0] if len(predictions) > 1 and predictions[1] else 0.0
    logger.debug(f"Language detection result: {lang_code} (confidence: {confidence:.2f})")
    print(f"Detected language: {lang_code}")
    logger.info(f"Language detection completed: {lang_code}")
    return {"detected_language": lang_code}

def translate_node(state: AgentState) -> dict:
    """Translates the transcript to English using the chosen LLM."""
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.chains import LLMChain

    logger.info("--- 📝 Translating to English ---")
    print("--- 📝 Translating to English ---")
    transcript = state.get("original_transcript")
    logger.debug(f"Original transcript length: {len(transcript) if transcript else 0} characters")
    if not transcript:
        logger.warning("No transcript provided for translation")
        print("No transcript available for translation.")
        return {"final_transcript": ""}
    
    logger.debug("Creating translation prompt template")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert translator. Translate the following text to English. Output only the translated text and nothing else."),
        HumanMessagePromptTemplate.from_template("{transcript}")
    ])
    logger.info("Initializing translation chain with LLM")
    translator_chain = LLMChain(prompt=prompt, llm=llm)
    logger.debug("Invoking translation chain")
    result = translator_chain.invoke({"transcript": transcript})
    translated_text = result["text"]
    logger.debug(f"Translated text length: {len(translated_text)} characters")
    print("Translation complete.")
    logger.info("Translation completed successfully")
    return {"final_transcript": translated_text}

def summarize_node(state: AgentState) -> dict:
    """Generates a structured summary of the final transcript."""
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.chains import LLMChain

    logger.info("--- 💼 Summarizing Transcript ---")
    print("--- 💼 Summarizing Transcript ---")
    transcript_to_summarize = state.get("final_transcript")
    logger.debug(f"Transcript to summarize length: {len(transcript_to_summarize) if transcript_to_summarize else 0} characters")
    if not transcript_to_summarize:
        logger.warning("No transcript provided for summarization")
        print("No transcript available for summarization.")
        return {"summary": "No transcript was available to summarize."}
    
    logger.debug("Creating summarization prompt template")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert in summarizing meetings. Provide a structured summary from the transcript with the following sections: Key Discussion Points, Action Items (with assigned owners if mentioned), and Key Decisions."),
        HumanMessagePromptTemplate.from_template("Please summarize the following meeting transcript:\n\n{transcript}")
    ])
    logger.info("Initializing summarization chain with LLM")
    summarizer_chain = LLMChain(prompt=prompt, llm=llm)
    logger.debug("Invoking summarization chain")
    result = summarizer_chain.invoke({"transcript": transcript_to_summarize})
    summary = result["text"]
    logger.debug(f"Summary length: {len(summary)} characters")
    print("Summary complete.")
    logger.info("Summarization completed successfully")
    return {"summary": summary}

# Conditional Edges

def route_after_language_detection(state: AgentState) -> str:
    """Decides whether to translate or go straight to summarization."""
    logger.info("--- 🚦 Routing based on language ---")
    print("--- 🚦 Routing based on language ---")
    lang_code = state.get("detected_language")
    logger.debug(f"Detected language code: {lang_code}")
    if lang_code == "en":
        logger.info("--- ✅ No translation needed ---")
        print("--- ✅ No translation needed ---")
        state["final_transcript"] = state["original_transcript"]
        logger.debug("Copied original transcript to final transcript")
        return "summarize"
    else:
        logger.info(f"--- ➡️ Translation required for language: {lang_code} ---")
        print("--- ➡️ Translation required ---")
        return "translate"

# Build and Compile the Graph

logger.info("Building and compiling the LangGraph workflow")
graph_builder = StateGraph(AgentState)
logger.debug("Adding nodes to graph")
graph_builder.add_node("transcribe", transcribe_audio_node)
graph_builder.add_node("detect_language", detect_language_node)
graph_builder.add_node("translate", translate_node)
graph_builder.add_node("summarize", summarize_node)
logger.debug("Setting entry point")
graph_builder.set_entry_point("transcribe")
logger.debug("Adding edges to graph")
graph_builder.add_edge("transcribe", "detect_language")
graph_builder.add_edge("translate", "summarize")
graph_builder.add_edge("summarize", END)
logger.debug("Adding conditional edges")
graph_builder.add_conditional_edges(
    "detect_language",
    route_after_language_detection,
    {"translate": "translate", "summarize": "summarize"},
)
logger.info("Compiling LangGraph workflow")
meeting_summarizer_app = graph_builder.compile()
logger.info("LangGraph workflow compiled successfully")