import os
import sys
import logging
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("meeting_summarizer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_console_app():
    """Run the original console application"""
    # Import here to avoid loading models if not needed
    from agents.summary_agent import meeting_summarizer_app
    
    logger.info("Starting Meeting Summarizer Application (Console Mode)")
    audio_file = "meeting_audio.mp3"
    
    logger.debug(f"Checking if audio file exists: {audio_file}")
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found at '{audio_file}'")
        print(f"Error: Audio file not found at '{audio_file}'.")
    else:
        logger.info(f"Audio file found: {audio_file}")
        initial_input = {"audio_file_path": audio_file}
        logger.debug(f"Invoking meeting summarizer app with input: {initial_input}")
        final_state = meeting_summarizer_app.invoke(initial_input)
        logger.debug("Meeting summarizer app execution completed")
        
        print("\n" + "="*50)
        print("          MEETING SUMMARY")
        print("="*50 + "\n")
        logger.debug("Displaying summary")
        print(final_state.get("summary"))
        print("\n" + "="*50)
        logger.info("Application finished successfully")

def run_ui():
    """Run the UI application"""
    # Check if tkinter is available
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Error: tkinter is not available in this environment.")
        print("Please install tkinter or run in console mode.")
        sys.exit(1)
    
    try:
        # Try to import the UI module
        spec = importlib.util.spec_from_file_location("ui", "ui.py")
        ui_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui_module)
        ui_module.main()
    except Exception as e:
        print(f"Error starting UI: {e}")
        print("Make sure you have all required dependencies installed.")
        sys.exit(1)

if __name__ == "__main__":
    # Check if UI mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        run_ui()
    else:
        run_console_app()