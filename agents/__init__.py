"""
HR Multi-Agent System - Agents Package

This package contains all the specialized agents for the HR recruitment workflow:
- CVParserAgent: Extracts information from CV files
- ScoringAgent: Scores candidates against job requirements
- ShortlistingAgent: Selects top candidates for review
- InterviewAgent: Generates tailored interview questions
- EmailAgent: Drafts personalized emails to candidates
"""

from .cv_parser_agent import CVParserAgent
from .scoring_agent import ScoringAgent
from .shortlisting_agent import ShortlistingAgent
from .interview_agent import InterviewAgent
from .email_agent import EmailAgent

__all__ = [
    'CVParserAgent',
    'ScoringAgent', 
    'ShortlistingAgent',
    'InterviewAgent',
    'EmailAgent'
] 