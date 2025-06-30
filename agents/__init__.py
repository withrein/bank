"""
HR Multi-Agent System - Agents Package

This package contains all the specialized agents for the HR recruitment workflow:
- CVAnalyzerAgent: Extracts information from CV files with bilingual support
- ScoringAgent: Scores candidates against job requirements with enhanced analysis
- ShortlistingAgent: Selects top candidates for review
- InterviewAgent: Generates tailored interview questions
- EmailAgent: Drafts personalized emails to candidates
- CoordinatorAgent: Orchestrates the entire workflow
"""

from .cv_parser_agent import CVAnalyzerAgent, CVParserAgent  # Backward compatibility
from .scoring_agent import ScoringAgent
from .shortlisting_agent import ShortlistingAgent
from .interview_agent import InterviewAgent
from .email_agent import EmailAgent

__all__ = [
    'CVAnalyzerAgent',
    'CVParserAgent',  # Backward compatibility
    'ScoringAgent', 
    'ShortlistingAgent',
    'InterviewAgent',
    'EmailAgent'
] 