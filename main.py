'''
Main script to run the AI Recruiter pipeline.
'''

from core.pipeline import RecruiterPipeline
from core.llm.gemini_llm import GeminiLLM
from core.tts.streaming_google_tts import StreamingGoogleTTS

def main():
    # Initialize components
    llm = GeminiLLM()
    tts = StreamingGoogleTTS()
    
    # Create and run pipeline
    pipeline = RecruiterPipeline(llm, tts)
    pipeline.run_conversation()

if __name__ == "__main__":
    main() 