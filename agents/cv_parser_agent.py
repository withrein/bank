import re
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import json

from models import ParsedCV, AgentState
from utils import (
    extract_text_from_file, clean_text, extract_email_from_text,
    extract_phone_from_text, extract_skills_from_text, 
    extract_years_of_experience, format_candidate_name
)
from config import Config

class CVParserAgent:
    """Agent responsible for parsing CV files and extracting structured information"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.get_gemini_api_key(),
            temperature=Config.TEMPERATURE
        )
        
    def parse_cv(self, file_path: str) -> ParsedCV:
        """Parse a single CV file and extract structured information"""
        try:
            # Extract raw text from file
            raw_text = extract_text_from_file(file_path)
            if not raw_text:
                raise ValueError(f"Could not extract text from {file_path}")
            
            # Clean the text
            cleaned_text = clean_text(raw_text)
            
            # Use LLM to extract structured information
            structured_data = self._extract_structured_data_with_llm(cleaned_text)
            
            # Extract basic information using regex patterns
            email = extract_email_from_text(raw_text)
            phone = extract_phone_from_text(raw_text)
            skills = extract_skills_from_text(raw_text)
            experience_years = extract_years_of_experience(raw_text)
            
            # Combine LLM results with regex extraction
            parsed_cv = ParsedCV(
                name=format_candidate_name(structured_data.get('name', 'Unknown Candidate')),
                email=email or structured_data.get('email'),
                phone=phone or structured_data.get('phone'),
                location=structured_data.get('location'),
                current_role=structured_data.get('current_role'),
                experience_years=experience_years or structured_data.get('experience_years'),
                skills=list(set(skills + structured_data.get('skills', []))),  # Combine and deduplicate
                education=structured_data.get('education', []),
                certifications=structured_data.get('certifications', []),
                work_experience=structured_data.get('work_experience', []),
                languages=structured_data.get('languages', []),
                summary=structured_data.get('summary'),
                raw_text=raw_text,
                file_name=file_path.split('/')[-1]
            )
            
            return parsed_cv
            
        except Exception as e:
            print(f"Error parsing CV {file_path}: {str(e)}")
            # Return minimal ParsedCV with error info
            return ParsedCV(
                name="Unknown Candidate",
                raw_text=f"Error parsing file: {str(e)}",
                file_name=file_path.split('/')[-1]
            )
    
    def _extract_structured_data_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract structured information from CV text"""
        
        system_prompt = """You are an expert CV parser. Extract structured information from the given CV text and return it as a JSON object.

Extract the following information:
- name: Full name of the candidate
- email: Email address
- phone: Phone number
- location: Current location/address
- current_role: Current or most recent job title
- experience_years: Total years of professional experience (as integer)
- skills: List of technical and professional skills
- education: List of educational qualifications with degree, institution, year
- certifications: List of professional certifications
- work_experience: List of work history with company, role, duration, responsibilities
- languages: List of languages spoken
- summary: Professional summary or objective

Return ONLY a valid JSON object. If information is not available, use null or empty array as appropriate.

Example format:
{
    "name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "+1-555-123-4567",
    "location": "New York, NY",
    "current_role": "Senior Software Engineer",
    "experience_years": 5,
    "skills": ["Python", "JavaScript", "React", "AWS"],
    "education": [
        {
            "degree": "Bachelor of Science in Computer Science",
            "institution": "University of Technology",
            "year": "2018"
        }
    ],
    "certifications": ["AWS Certified Solutions Architect"],
    "work_experience": [
        {
            "company": "Tech Corp",
            "role": "Software Engineer",
            "duration": "2020-2023",
            "responsibilities": ["Developed web applications", "Led team of 3 developers"]
        }
    ],
    "languages": ["English", "Spanish"],
    "summary": "Experienced software engineer with expertise in full-stack development"
}"""

        human_prompt = f"Parse the following CV text and extract structured information:\n\n{text}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                # Try to parse the entire response as JSON
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON response: {str(e)}")
            return {}
        except Exception as e:
            print(f"Error with LLM extraction: {str(e)}")
            return {}
    
    def parse_multiple_cvs(self, file_paths: List[str]) -> List[ParsedCV]:
        """Parse multiple CV files"""
        parsed_cvs = []
        
        for file_path in file_paths:
            print(f"Parsing CV: {file_path}")
            parsed_cv = self.parse_cv(file_path)
            parsed_cvs.append(parsed_cv)
        
        return parsed_cvs
    
    def process(self, state: AgentState) -> AgentState:
        """Process CV files in the agent state"""
        if not state.cv_files:
            state.errors.append("No CV files provided for parsing")
            return state
        
        try:
            print("🔍 CV Parser Agent: Starting CV parsing...")
            state.current_step = "parsing_cvs"
            
            # Parse all CV files
            parsed_cvs = self.parse_multiple_cvs(state.cv_files)
            state.parsed_cvs = parsed_cvs
            
            print(f"✅ CV Parser Agent: Successfully parsed {len(parsed_cvs)} CVs")
            
            # Log parsing results
            for cv in parsed_cvs:
                print(f"   - {cv.name} ({cv.file_name})")
                if cv.skills:
                    print(f"     Skills: {', '.join(cv.skills[:5])}{'...' if len(cv.skills) > 5 else ''}")
                if cv.experience_years:
                    print(f"     Experience: {cv.experience_years} years")
            
            state.current_step = "cvs_parsed"
            
        except Exception as e:
            error_msg = f"CV Parser Agent error: {str(e)}"
            print(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 