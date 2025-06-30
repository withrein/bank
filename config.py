import os
from typing import Optional

class Config:
    """Configuration class for the HR Multi-Agent System"""
    
    # API Keys
    GEMINI_API_KEY = "AIzaSyCjgLzf0RICqcGegobUT4M38ZZAeymHG2U"
    OPENAI_API_KEY = "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"
    
    # Application Settings
    MAX_CANDIDATES_TO_SHORTLIST = 5
    MINIMUM_SCORE_THRESHOLD = 60
    
    # File Upload Settings
    ALLOWED_CV_FORMATS = ['.pdf', '.docx', '.doc', '.txt']
    MAX_FILE_SIZE_MB = 10
    
    # Model Settings - Updated to use GPT-4o
    MODEL_PROVIDER = "openai"  # Changed from gemini to openai
    OPENAI_MODEL = "gpt-4o"  # Updated to GPT-4o
    GEMINI_MODEL = "gemini-1.5-flash"  # Keep for backward compatibility
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096  # Increased for better context
    
    # Language Settings
    # Interface is in Mongolian, but all AI generation in English
    INTERFACE_LANGUAGE = "mn"  # Streamlit interface in Mongolian
    AI_OUTPUT_LANGUAGE = "en"  # All AI generation in English only
    ENABLE_MONGOLIAN_TRANSLATION = False  # No translation needed
    
    # Enhanced Prompt Engineering Settings
    PROMPT_ENGINEERING = {
        "system_context": {
            "role": "You are an expert HR specialist with deep knowledge in recruitment, candidate evaluation, and professional communication.",
            "expertise": [
                "CV/Resume analysis and parsing",
                "Skills assessment and matching",
                "Interview question design",
                "Professional email drafting",
                "Candidate evaluation and ranking"
            ],
            "input_languages": ["English", "Mongolian (Cyrillic)"],
            "output_language": "English",
            "tone": "Professional, analytical, and culturally sensitive",
            "instruction": "Always generate responses in English, even when processing Mongolian input documents."
        },
        "agent_instructions": {
            "cv_parser": {
                "focus": "Extract structured information with high accuracy from both English and Mongolian CVs",
                "attention": ["Contact details", "Work experience", "Skills", "Education", "Certifications"],
                "output_format": "Structured JSON with confidence scores",
                "output_language": "All extracted information and analysis in English"
            },
            "scoring_agent": {
                "methodology": "Multi-dimensional scoring with detailed justification",
                "criteria": ["Skills match", "Experience relevance", "Education fit", "Cultural alignment"],
                "scoring_scale": "0-100 with detailed breakdown and reasoning",
                "output_language": "All scoring, recommendations, and analysis in English"
            },
            "interview_agent": {
                "question_types": ["Technical", "Behavioral", "Situational", "Role-specific"],
                "difficulty_levels": ["Entry", "Intermediate", "Senior", "Expert"],
                "customization": "Tailor questions to candidate background and job requirements",
                "output_language": "All interview questions and instructions in English"
            },
            "email_agent": {
                "templates": ["Interview invitation", "Rejection", "Follow-up", "Acknowledgment"],
                "personalization": "Include candidate name, specific qualifications, and relevant details",
                "tone": "Professional yet warm and encouraging",
                "output_language": "All email content generated in English"
            }
        }
    }
    
    # Model Context Protocol Settings
    MODEL_CONTEXT = {
        "context_window": 32768,  # GPT-4o context window
        "conversation_memory": True,
        "agent_memory": {
            "short_term": 10,  # Last 10 interactions
            "long_term": 100,  # Store key insights
            "cross_agent": True  # Share context between agents
        },
        "context_compression": {
            "enabled": True,
            "summarization_threshold": 20000,  # Compress when context exceeds this
            "key_information_retention": ["Job requirements", "Candidate profiles", "Scoring criteria"]
        }
    }
    
    # Bilingual Support Settings
    DEFAULT_LANGUAGE = "en"  # All AI output in English
    SUPPORTED_CV_LANGUAGES = ["mn", "en"]  # Can parse both Mongolian and English CVs
    INTERFACE_LANGUAGE_DEFAULT = "mn"  # Interface defaults to Mongolian
    
    # Mongolian Language Processing
    MONGOLIAN_KEYWORDS = {
        "education": ["сургууль", "их сургууль", "коллеж", "университет", "боловсрол", "диплом", "зэрэг"],
        "experience": ["ажлын туршлага", "туршлага", "ажил", "албан тушаал", "компани", "байгууллага"],
        "skills": ["чадвар", "ур чадвар", "мэдлэг", "технологи", "хэрэгсэл", "програм"],
        "languages": ["хэл", "хэлний чадвар", "англи хэл", "монгол хэл", "орос хэл", "хятад хэл"],
        "certifications": ["гэрчилгээ", "сертификат", "мэргэшил", "зэрэг цол"],
        "contact": ["холбоо барих", "утас", "имэйл", "хаяг", "байршил"]
    }
    
    # Scoring Weights
    SCORING_WEIGHTS = {
        "experience": 0.40,  # 40%
        "education": 0.25,   # 25%
        "skills": 0.25,      # 25%
        "other": 0.10        # 10%
    }
    
    # Email Templates Language Support
    EMAIL_LANGUAGES = {
        "mn": {
            "interview_invitation_subject": "{company}-д {job_title} албан тушаалд ярилцлагад урих",
            "rejection_subject": "{company}-ээс хариу",
            "acknowledgment_subject": "Таны өргөдлийг хүлээн авлаа"
        },
        "en": {
            "interview_invitation_subject": "Interview Invitation - {job_title} at {company}",
            "rejection_subject": "Application Update from {company}",
            "acknowledgment_subject": "Application Received - {company}"
        }
    }
    
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key from environment or config"""
        return os.getenv("GEMINI_API_KEY", cls.GEMINI_API_KEY)
    
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment or config"""
        return os.getenv("OPENAI_API_KEY", cls.OPENAI_API_KEY)
    
    @classmethod
    def get_current_model_config(cls) -> dict:
        """Get current model configuration based on provider"""
        if cls.MODEL_PROVIDER == "openai":
            return {
                "provider": "openai",
                "model": cls.OPENAI_MODEL,
                "api_key": cls.get_openai_api_key(),
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS
            }
        else:
            return {
                "provider": "gemini", 
                "model": cls.GEMINI_MODEL,
                "api_key": cls.get_gemini_api_key(),
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS
            }
    
    @classmethod
    def get_language_keywords(cls, language: str = "mn") -> dict:
        """Get language-specific keywords for CV parsing"""
        if language == "mn":
            return cls.MONGOLIAN_KEYWORDS
        else:
            # English keywords (default patterns)
            return {
                "education": ["education", "school", "university", "college", "degree", "diploma"],
                "experience": ["experience", "work", "job", "position", "company", "organization"],
                "skills": ["skills", "abilities", "knowledge", "technology", "tools", "software"],
                "languages": ["languages", "language skills", "english", "mongolian", "chinese", "russian"],
                "certifications": ["certification", "certificate", "qualification", "credential"],
                "contact": ["contact", "phone", "email", "address", "location"]
            } 