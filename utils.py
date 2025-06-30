import os
import re
import PyPDF2
import pdfplumber
from docx import Document
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using multiple methods for better accuracy"""
    text = ""
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If pdfplumber didn't extract much text, try PyPDF2
        if len(text.strip()) < 100:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""
    
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error extracting text from TXT {file_path}: {str(e)}")
            return ""
    except Exception as e:
        print(f"Error extracting text from TXT {file_path}: {str(e)}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats"""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\@\+\#]', ' ', text)
    
    # Remove excessive spaces again after character removal
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text"""
    # Various phone number patterns
    phone_patterns = [
        r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',  # US format
        r'\+?(\d{1,3})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})[-.\s]?(\d{3,4})',  # International
        r'(\d{10})',  # 10 digits
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                return ''.join(matches[0])
            else:
                return matches[0]
    
    return None

def extract_skills_from_text(text: str, skill_keywords: List[str] = None) -> List[str]:
    """Extract skills from text based on common keywords"""
    if skill_keywords is None:
        # Common technical skills keywords
        skill_keywords = [
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
            'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring',
            'laravel', 'rails', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd',
            'terraform', 'ansible',
            
            # Other Technical Skills
            'machine learning', 'data science', 'artificial intelligence', 'blockchain',
            'cybersecurity', 'network security', 'web development', 'mobile development',
            'ui/ux design', 'product management', 'project management', 'agile', 'scrum'
        ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))  # Remove duplicates

def extract_years_of_experience(text: str) -> Optional[int]:
    """Extract years of experience from text"""
    # Patterns to match experience mentions
    patterns = [
        r'(\d+)\s*(?:\+)?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\s*(?:\+)?\s*years?',
        r'(\d+)\s*(?:\+)?\s*yrs?\s*(?:of)?\s*experience',
        r'(\d+)\s*(?:\+)?\s*years?\s*in',
    ]
    
    text_lower = text.lower()
    max_years = 0
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                years = int(match)
                if years > max_years and years <= 50:  # Reasonable upper limit
                    max_years = years
            except ValueError:
                continue
    
    return max_years if max_years > 0 else None

def save_json_output(data: Any, file_path: str) -> None:
    """Save data as JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if hasattr(data, 'model_dump'):
                json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)
            elif hasattr(data, 'dict'):
                json.dump(data.dict(), f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {str(e)}")

def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {str(e)}")
        return None

def validate_file_format(file_path: str, allowed_formats: List[str]) -> bool:
    """Validate if file format is allowed"""
    file_extension = Path(file_path).suffix.lower()
    return file_extension in allowed_formats

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0

def create_output_directory(base_path: str = "outputs") -> str:
    """Create output directory if it doesn't exist"""
    os.makedirs(base_path, exist_ok=True)
    return base_path

def format_candidate_name(name: str) -> str:
    """Format candidate name for consistency"""
    if not name:
        return "Unknown Candidate"
    
    # Remove extra whitespace and title case
    name = ' '.join(name.split())
    return name.title()

def calculate_skill_match_percentage(candidate_skills: List[str], required_skills: List[str]) -> float:
    """Calculate percentage of skill match"""
    if not required_skills:
        return 100.0
    
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    required_skills_lower = [skill.lower() for skill in required_skills]
    
    matched_skills = set(candidate_skills_lower) & set(required_skills_lower)
    
    return (len(matched_skills) / len(required_skills_lower)) * 100 