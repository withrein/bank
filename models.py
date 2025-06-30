from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class CandidateStatus(str, Enum):
    """Enum for candidate status"""
    PENDING = "pending"
    SHORTLISTED = "shortlisted"
    REJECTED = "rejected"
    INTERVIEWED = "interviewed"

class ParsedCV(BaseModel):
    """Model for parsed CV data"""
    name: str = Field(..., description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    
    # Professional Information
    current_role: Optional[str] = Field(None, description="Current job title")
    experience_years: Optional[int] = Field(None, description="Total years of experience")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    
    # Education
    education: List[Dict[str, Any]] = Field(default_factory=list, description="Educational background")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    
    # Work Experience
    work_experience: List[Dict[str, Any]] = Field(default_factory=list, description="Work history")
    
    # Additional Information
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    summary: Optional[str] = Field(None, description="Professional summary")
    
    # Metadata
    raw_text: str = Field(..., description="Raw extracted text from CV")
    file_name: str = Field(..., description="Original file name")

class JobDescription(BaseModel):
    """Model for job description"""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    
    # Requirements
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_experience: Optional[int] = Field(None, description="Minimum years of experience")
    education_requirements: List[str] = Field(default_factory=list, description="Education requirements")
    
    # Job Details
    job_type: Optional[str] = Field(None, description="Full-time, Part-time, Contract, etc.")
    salary_range: Optional[str] = Field(None, description="Salary range if provided")
    description: str = Field(..., description="Full job description text")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")

class CandidateScore(BaseModel):
    """Model for candidate scoring"""
    candidate_name: str = Field(..., description="Candidate name")
    file_name: str = Field(..., description="CV file name")
    
    # Scoring breakdown
    skills_match_score: float = Field(..., ge=0, le=100, description="Skills matching score (0-100)")
    experience_score: float = Field(..., ge=0, le=100, description="Experience score (0-100)")
    education_score: float = Field(..., ge=0, le=100, description="Education score (0-100)")
    overall_score: float = Field(..., ge=0, le=100, description="Overall matching score (0-100)")
    
    # Detailed analysis
    matched_skills: List[str] = Field(default_factory=list, description="Skills that match job requirements")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills not found in CV")
    strengths: List[str] = Field(default_factory=list, description="Candidate strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    
    # Recommendation
    recommendation: str = Field(..., description="Hiring recommendation")
    reasoning: str = Field(..., description="Detailed reasoning for the score")

class InterviewQuestion(BaseModel):
    """Model for interview questions"""
    question: str = Field(..., description="The interview question")
    category: str = Field(..., description="Question category (technical, behavioral, etc.)")
    difficulty: str = Field(..., description="Question difficulty (easy, medium, hard)")
    expected_answer_points: List[str] = Field(default_factory=list, description="Key points for good answers")

class CandidateQuestions(BaseModel):
    """Model for candidate-specific interview questions"""
    candidate_name: str = Field(..., description="Candidate name")
    job_title: str = Field(..., description="Job title")
    
    technical_questions: List[InterviewQuestion] = Field(default_factory=list)
    behavioral_questions: List[InterviewQuestion] = Field(default_factory=list)
    role_specific_questions: List[InterviewQuestion] = Field(default_factory=list)
    
    total_questions: int = Field(default=0, description="Total number of questions generated")

class EmailDraft(BaseModel):
    """Model for email drafts"""
    recipient_name: str = Field(..., description="Recipient name")
    recipient_email: str = Field(..., description="Recipient email")
    email_type: str = Field(..., description="Type of email (invitation, rejection, etc.)")
    
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    
    # Email metadata
    job_title: str = Field(..., description="Job title")
    company_name: str = Field(..., description="Company name")
    interview_date: Optional[str] = Field(None, description="Interview date if applicable")
    interview_time: Optional[str] = Field(None, description="Interview time if applicable")

class AgentState(BaseModel):
    """Model for agent workflow state"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    job_description: Optional[JobDescription] = None
    cv_files: List[str] = Field(default_factory=list, description="List of CV file paths")
    parsed_cvs: List[ParsedCV] = Field(default_factory=list)
    candidate_scores: List[CandidateScore] = Field(default_factory=list)
    shortlisted_candidates: List[CandidateScore] = Field(default_factory=list)
    interview_questions: Dict[str, CandidateQuestions] = Field(default_factory=dict)
    email_drafts: List[EmailDraft] = Field(default_factory=list)
    
    # Workflow metadata
    current_step: str = Field(default="start", description="Current workflow step")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    processing_status: str = Field(default="pending", description="Overall processing status") 