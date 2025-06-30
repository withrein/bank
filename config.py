import os
from typing import Optional

class Config:
    """Configuration class for the HR Multi-Agent System"""
    
    # API Keys
    GEMINI_API_KEY = "AIzaSyAmbfwjH7DzC9Rd4uZuFUJ5O3_bE00Lalw"
    OPENAI_API_KEY = "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"
    
    # Application Settings
    MAX_CANDIDATES_TO_SHORTLIST = 5
    MINIMUM_SCORE_THRESHOLD = 60
    
    # File Upload Settings
    ALLOWED_CV_FORMATS = ['.pdf', '.docx', '.doc', '.txt']
    MAX_FILE_SIZE_MB = 10
    
    # Model Settings
    GEMINI_MODEL = "gemini-1.5-flash"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key from environment or config"""
        return os.getenv("GEMINI_API_KEY", cls.GEMINI_API_KEY)
    
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment or config"""
        return os.getenv("OPENAI_API_KEY", cls.OPENAI_API_KEY) 