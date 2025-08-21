import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import numpy as np
from scipy.io.wavfile import write
import time
from dotenv import set_key, get_key
import importlib.util

# Cross-platform audio recording
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    sd = None
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio recording will be disabled.")

# Try to import the required modules
try:
    spec = importlib.util.spec_from_file_location("audio_export", "audio_export.py")
    audio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_module)
    record_audio_function = audio_module.record_audio
except Exception as e:
    print(f"Warning: Could not import audio_export module: {e}")
    record_audio_function = None

try:
    spec = importlib.util.spec_from_file_location("summary_agent", "agents/summary_agent.py")
    summary_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary_module)
    meeting_summarizer_app = summary_module.meeting_summarizer_app
except Exception as e:
    print(f"Error: Could not import summary_agent module: {e}")
    meeting_summarizer_app = None

class MeetingSummarizerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Summarizer")
        self.root.geometry("800x700")
        
        # Handle Windows scaling
        if sys.platform == "win32":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass
        
        # Variables
        self.recording = False
        self.recorded_frames = []
        self.sample_rate = 44100
        self.output_filename = "meeting_audio.wav"
        self.api_key = ""
        self.llm_provider = tk.StringVar(value="openai")
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # LLM Provider Selection
        ttk.Label(main_frame, text="LLM Provider:").grid(row=0, column=0, sticky=tk.W, pady=5)
        provider_frame = ttk.Frame(main_frame)
        provider_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(provider_frame, text="OpenAI", variable=self.llm_provider, 
                       value="openai").pack(side=tk.LEFT)
        ttk.Radiobutton(provider_frame, text="Google Gemini", variable=self.llm_provider, 
                       value="gemini").pack(side=tk.LEFT, padx=(10, 0))
        
        # API Key Input
        ttk.Label(main_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.api_key_entry = ttk.Entry(main_frame, width=50, show="*")
        self.api_key_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        self.api_key_entry.bind("<FocusOut>", self.save_api_key)
        
        # Load saved API key if exists
        self.load_api_key()
        
        # Audio Recording Controls (only if audio is available)
        if AUDIO_AVAILABLE:
            ttk.Label(main_frame, text="Audio Recording:", font=("Arial", 12, "bold")).grid(
                row=2, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
            
            record_frame = ttk.Frame(main_frame)
            record_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            self.record_button = ttk.Button(record_frame, text="Start Recording", 
                                           command=self.toggle_recording)
            self.record_button.pack(side=tk.LEFT)
            
            self.recording_status = ttk.Label(record_frame, text="Not recording")
            self.recording_status.pack(side=tk.LEFT, padx=(10, 0))
        else:
            ttk.Label(main_frame, text="Audio Recording: Not Available", font=("Arial", 12, "bold"), foreground="red").grid(
                row=2, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
            ttk.Label(main_frame, text="Install sounddevice to enable recording: pip install sounddevice").grid(
                row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Audio File Selection (Alternative to recording)
        ttk.Label(main_frame, text="Or select existing audio file:").grid(
            row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=(5, 0))
        
        # Summarize Button
        self.summarize_button = ttk.Button(main_frame, text="Summarize Meeting", 
                                          command=self.summarize_meeting, state=tk.DISABLED)
        self.summarize_button.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Summary Display
        ttk.Label(main_frame, text="Meeting Summary:", font=("Arial", 12, "bold")).grid(
            row=7, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
        
        # Text widget with scrollbar for summary
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        self.summary_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=70, height=15)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Export Button
        self.export_button = ttk.Button(main_frame, text="Export as PDF", 
                                       command=self.export_pdf, state=tk.DISABLED)
        self.export_button.grid(row=9, column=0, columnspan=2, pady=20)
        
    def load_api_key(self):
        """Load saved API key from .env file"""
        if os.path.exists(".env"):
            try:
                # Try to get the key based on current provider
                provider = self.llm_provider.get()
                key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
                
                # For cross-platform compatibility
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith(key_name + "="):
                            self.api_key_entry.insert(0, line.split("=", 1)[1].strip())
                            break
            except Exception as e:
                print(f"Error loading API key: {e}")
    
    def save_api_key(self, event=None):
        """Save API key to .env file"""
        self.api_key = self.api_key_entry.get()
        provider = self.llm_provider.get()
        
        if self.api_key:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
            set_key(".env", key_name, self.api_key)
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Recording Error", "Audio recording is not available. Please install sounddevice.")
            return
            
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        if not AUDIO_AVAILABLE:
            return
            
        if self.recording:
            return
            
        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.recording_status.config(text="Recording...")
        self.recorded_frames = []
        self.summarize_button.config(state=tk.DISABLED)
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop audio recording"""
        if not AUDIO_AVAILABLE:
            return
            
        self.recording = False
        self.record_button.config(text="Start Recording")
        self.recording_status.config(text=f"Saved as {self.output_filename}")
        self.summarize_button.config(state=tk.NORMAL)
        
        # Save the recording
        if self.recorded_frames:
            recording = np.concatenate(self.recorded_frames, axis=0)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.output_filename) if os.path.dirname(self.output_filename) else ".", exist_ok=True)
            write(self.output_filename, self.sample_rate, (recording * 32767).astype(np.int16))
    
    def record_audio(self):
        """Record audio in a separate thread"""
        if not AUDIO_AVAILABLE:
            return
            
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                self.recorded_frames.append(indata.copy())
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=2, callback=audio_callback):
                while self.recording:
                    time.sleep(0.1)
        except Exception as e:
            messagebox.showerror("Recording Error", f"An error occurred during recording: {str(e)}")
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.recording_status.config(text="Error in recording")
    
    def browse_file(self):
        """Browse for an existing audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if file_path:
            # Normalize path for Windows
            file_path = os.path.normpath(file_path)
            self.file_path_var.set(file_path)
            self.summarize_button.config(state=tk.NORMAL)
    
    def summarize_meeting(self):
        """Summarize the meeting"""
        # Get the audio file path
        file_path = self.file_path_var.get()
        if not file_path:
            if AUDIO_AVAILABLE:
                file_path = self.output_filename
            else:
                messagebox.showerror("File Error", "Please select an audio file.")
                return
                
        # Normalize path for Windows
        file_path = os.path.normpath(file_path)
            
        if not os.path.exists(file_path):
            messagebox.showerror("File Error", f"Audio file not found: {file_path}")
            return
            
        # Save API key
        self.save_api_key()
        
        # Update UI
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Processing audio file and generating summary...\n")
        self.summary_text.insert(tk.END, "This may take a few moments...\n\n")
        self.summarize_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        
        # Process in a separate thread
        threading.Thread(target=self.process_summary, args=(file_path,), daemon=True).start()
    
    def process_summary(self, audio_file_path):
        """Process the summary in a separate thread"""
        try:
            # Update the LLM provider in the summary module
            if hasattr(summary_module, 'LLM_PROVIDER'):
                summary_module.LLM_PROVIDER = self.llm_provider.get()
            
            # Reload the module to apply changes
            importlib.reload(summary_module)
            
            # Run the summarization
            initial_input = {"audio_file_path": audio_file_path}
            final_state = summary_module.meeting_summarizer_app.invoke(initial_input)
            summary = final_state.get("summary", "No summary generated.")
            
            # Update UI in the main thread
            self.root.after(0, self.display_summary, summary)
        except Exception as e:
            error_msg = f"Error during summarization: {str(e)}"
            self.root.after(0, self.display_summary, error_msg)
    
    def display_summary(self, summary):
        """Display the summary in the text widget"""
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summarize_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)
    
    def export_pdf(self):
        """Export summary as PDF"""
        try:
            # Try to import fpdf for PDF generation
            from fpdf import FPDF
        except ImportError:
            messagebox.showinfo("PDF Export", 
                "PDF export requires the fpdf library.\n"
                "Install it with: pip install fpdf")
            return
        
        # Get summary text
        summary_text = self.summary_text.get(1.0, tk.END).strip()
        if not summary_text:
            messagebox.showwarning("Export Error", "No summary to export.")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Summary as PDF"
        )
        
        if file_path:
            # Normalize path for Windows
            file_path = os.path.normpath(file_path)
            
            try:
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                
                # Add title
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Meeting Summary", ln=True, align="C")
                pdf.ln(10)
                
                # Add summary text
                pdf.set_font("Arial", size=12)
                # Split text into lines that fit the page
                for line in summary_text.split('\n'):
                    pdf.multi_cell(0, 10, line)
                
                # Save PDF
                pdf.output(file_path)
                messagebox.showinfo("Export Success", f"Summary exported successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export PDF: {str(e)}")

def main():
    root = tk.Tk()
    app = MeetingSummarizerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()