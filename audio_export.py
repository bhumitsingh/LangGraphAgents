import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import sys
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_export.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLE_RATE = 44100  # Standard sample rate for audio
OUTPUT_FILENAME = "meeting_audio.wav" # The file to save the recording to

def record_audio():
    """
    Records audio from the default microphone and saves it to a WAV file.
    This function is intended for standalone use and will block until Ctrl+C is pressed.
    """
    logger.info("Starting audio recording function")
    print("🎙️  Starting audio recording...")
    print("Press Ctrl+C to stop the recording.")
    
    recorded_frames = []
    
    logger.debug(f"Sample rate: {SAMPLE_RATE}")
    logger.debug(f"Output filename: {OUTPUT_FILENAME}")

    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"Audio status: {status}")
            print(status, file=sys.stderr)
        logger.debug(f"Recording audio chunk with {frames} frames")
        recorded_frames.append(indata.copy())

    try:
        logger.info("Initializing audio input stream")
        # Use a stream to continuously record audio data
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=2, callback=audio_callback):
            logger.info("Audio stream started, recording in progress")
            while True:
                # The recording happens in the background via the callback
                # We can sleep here to keep the main thread alive
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Recording stopped by user (Ctrl+C)")
        print("\n🛑 Recording stopped.")
        
        if not recorded_frames:
            logger.warning("No audio was recorded")
            print("No audio was recorded.")
            return False

        logger.info(f"Saving recording to '{OUTPUT_FILENAME}'")
        print(f"💾 Saving recording to '{OUTPUT_FILENAME}'...")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILENAME) if os.path.dirname(OUTPUT_FILENAME) else ".", exist_ok=True)
        
        # Concatenate all recorded frames into a single NumPy array
        logger.debug(f"Concatenating {len(recorded_frames)} recorded frames")
        recording = np.concatenate(recorded_frames, axis=0)
        logger.debug(f"Recording shape: {recording.shape}")
        
        # Save the NumPy array as a WAV file
        # Note: We use np.int16 because it's a standard format for WAV files.
        logger.info("Writing audio data to WAV file")
        write(OUTPUT_FILENAME, SAMPLE_RATE, (recording * 32767).astype(np.int16))
        
        logger.info(f"Successfully saved audio to '{OUTPUT_FILENAME}'")
        print(f"✅ Successfully saved audio to '{OUTPUT_FILENAME}'.")
        print("You can now run the summarizer agent with this file.")
        return True

    except Exception as e:
        logger.error(f"An error occurred during recording: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        return False

def record_audio_frames(sample_rate=44100):
    """
    Records audio and returns the recorded frames.
    This function is intended for use in UI applications where custom control is needed.
    
    Args:
        sample_rate (int): The sample rate for recording
        
    Returns:
        list: List of numpy arrays containing the recorded audio frames
    """
    recorded_frames = []
    
    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"Audio status: {status}")
        recorded_frames.append(indata.copy())
    
    try:
        # Use a stream to continuously record audio data
        with sd.InputStream(samplerate=sample_rate, channels=2, callback=audio_callback):
            # This function is meant to be used in a context where the caller controls the duration
            # The caller should break out of the recording loop when needed
            pass
            
    except Exception as e:
        logger.error(f"An error occurred during recording: {e}", exc_info=True)
    
    return recorded_frames