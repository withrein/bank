import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import unicodedata

from models import ParsedCV, AgentState
from utils import (
    extract_text_from_file, clean_text, extract_email_from_text,
    extract_phone_from_text, extract_skills_from_text, 
    extract_years_of_experience, format_candidate_name
)
from config import Config
from .base_agent import EnhancedBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVParserAgent(EnhancedBaseAgent):
    """Enhanced CV Analyzer Agent with bilingual support (Mongolian/English)"""
    
    def __init__(self):
        super().__init__("cv_parser")
        self.mongolian_keywords = Config.get_language_keywords("mn")
        self.english_keywords = Config.get_language_keywords("en")
        
    def detect_language(self, text: str) -> str:
        """Detect if the CV is primarily in Mongolian or English"""
        if not text:
            return "en"
        
        # Count Cyrillic characters for Mongolian detection
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text if char.isalpha() and ord(char) < 256)
        
        # If more than 30% of alphabetic characters are Cyrillic, consider it Mongolian
        total_alpha = cyrillic_count + latin_count
        if total_alpha > 0 and (cyrillic_count / total_alpha) > 0.3:
            return "mn"
        
        # Check for Mongolian keywords
        mongolian_keywords_found = sum(1 for keyword_list in self.mongolian_keywords.values() 
                                     for keyword in keyword_list if keyword.lower() in text.lower())
        
        if mongolian_keywords_found >= 3:
            return "mn"
        
        return "en"
    
    def normalize_mongolian_text(self, text: str) -> str:
        """Normalize Mongolian Cyrillic text for better processing"""
        if not text:
            return text
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Common Mongolian text cleaning
        replacements = {
            'ө': 'ө', 'ү': 'ү', 'ё': 'ё',  # Normalize special characters
            '  ': ' ',  # Multiple spaces to single space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Enhanced PDF text extraction with encoding handling"""
        try:
            raw_text = extract_text_from_file(file_path)
            if raw_text:
                # Handle encoding issues for Cyrillic text
                try:
                    # Try UTF-8 first
                    raw_text = raw_text.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    # Fallback to other encodings
                    try:
                        raw_text = raw_text.encode('cp1251').decode('cp1251')
                    except UnicodeError:
                        logger.warning(f"Encoding issues detected in {file_path}")
                
                return self.normalize_mongolian_text(raw_text)
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Enhanced DOCX text extraction with encoding handling"""
        try:
            raw_text = extract_text_from_file(file_path)
            if raw_text:
                return self.normalize_mongolian_text(raw_text)
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    def parse_personal_info(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Extract personal information with language-specific patterns"""
        info = {}
        
        # Extract email (universal pattern)
        info['email'] = extract_email_from_text(text)
        
        # Extract phone (enhanced patterns for Mongolian formats)
        phone_patterns = [
            r'(?:утас|гар утас|phone)?\s*:?\s*(\+976\s?\d{8})',  # Mongolian mobile format
            r'(?:утас|гар утас|phone)?\s*:?\s*(\d{8})',          # 8-digit format
            r'(?:phone|tel|утас)?\s*:?\s*(\+?[\d\s\-\(\)]{7,15})'  # General format
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['phone'] = match.group(1).strip()
                break
        
        # Extract name (different patterns for different languages)
        if language == "mn":
            # Mongolian name patterns
            name_patterns = [
                r'(?:Нэр|нэр|Овог нэр|овог нэр)\s*:?\s*([А-Яа-яЁёӨөҮү\s]{2,50})',
                r'^([А-Яа-яЁёӨөҮү]{2,20}\s+[А-Яа-яЁёӨөҮү]{2,20})',  # First line name
            ]
        else:
            # English name patterns
            name_patterns = [
                r'(?:Name|name|Full Name|full name)\s*:?\s*([A-Za-z\s]{2,50})',
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+)',  # Capitalized first and last name
            ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                info['name'] = match.group(1).strip()
                break
        
        # Extract location
        location_keywords = self.mongolian_keywords['contact'] if language == "mn" else self.english_keywords['contact']
        location_pattern = rf"(?:{'|'.join(location_keywords)})\s*:?\s*([^,\n]+)"
        location_match = re.search(location_pattern, text, re.IGNORECASE)
        if location_match:
            info['location'] = location_match.group(1).strip()
        
        return info
    
    def parse_education(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """Extract education information with language-specific patterns"""
        education_list = []
        
        keywords = self.mongolian_keywords['education'] if language == "mn" else self.english_keywords['education']
        
        # Find education sections
        education_pattern = rf"(?:{'|'.join(keywords)})[^а-яё]*?(\d{{4}}|\d{{2}}-\d{{2}})"
        
        matches = re.finditer(education_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            section = match.group(0)
            education_info = {}
            
            # Extract year
            year_match = re.search(r'(\d{4})', section)
            if year_match:
                education_info['year'] = year_match.group(1)
            
            # Extract institution and degree (basic extraction)
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Reasonable length for institution/degree
                    if 'institution' not in education_info:
                        education_info['institution'] = line
                    elif 'degree' not in education_info and line != education_info.get('institution'):
                        education_info['degree'] = line
            
            if education_info:
                education_list.append(education_info)
        
        return education_list
    
    def parse_experience(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """Extract work experience with language-specific patterns"""
        experience_list = []
        
        keywords = self.mongolian_keywords['experience'] if language == "mn" else self.english_keywords['experience']
        
        # Look for experience sections
        experience_pattern = rf"(?:{'|'.join(keywords)})"
        
        # Split text into potential experience sections
        sections = re.split(experience_pattern, text, flags=re.IGNORECASE)
        
        for section in sections[1:]:  # Skip first section (before first keyword)
            if len(section.strip()) < 50:  # Too short to be meaningful
                continue
            
            experience_info = {}
            
            # Extract years (various formats)
            year_patterns = [
                r'(\d{4})\s*[-–—]\s*(\d{4})',  # 2020-2023
                r'(\d{4})\s*[-–—]\s*(?:одоо|present|current)',  # 2020-present
                r'(\d{1,2})/(\d{4})\s*[-–—]\s*(\d{1,2})/(\d{4})',  # MM/YYYY format
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        experience_info['start_year'] = match.group(1)
                        experience_info['end_year'] = match.group(2)
                    elif len(match.groups()) == 4:
                        experience_info['start_date'] = f"{match.group(1)}/{match.group(2)}"
                        experience_info['end_date'] = f"{match.group(3)}/{match.group(4)}"
                    break
            
            # Extract company and role (first few meaningful lines)
            lines = [line.strip() for line in section.split('\n') if line.strip() and len(line.strip()) > 5]
            
            if lines:
                if len(lines) >= 1:
                    experience_info['company'] = lines[0]
                if len(lines) >= 2:
                    experience_info['role'] = lines[1]
                if len(lines) >= 3:
                    experience_info['responsibilities'] = lines[2:5]  # Take next 3 lines as responsibilities
            
            if experience_info:
                experience_list.append(experience_info)
        
        return experience_list
    
    def parse_skills(self, text: str, language: str = "en") -> List[str]:
        """Extract skills with enhanced patterns for both languages"""
        skills = set()
        
        # Use utility function for basic skill extraction
        basic_skills = extract_skills_from_text(text)
        skills.update(basic_skills)
        
        # Language-specific skill extraction
        skill_keywords = self.mongolian_keywords['skills'] if language == "mn" else self.english_keywords['skills']
        
        # Find skills sections
        for keyword in skill_keywords:
            pattern = rf"{keyword}\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\n[A-ZА-Я]|$)"
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                skills_text = match.group(1)
                # Split by common delimiters
                skill_items = re.split(r'[,;•\-\n]', skills_text)
                for item in skill_items:
                    item = item.strip()
                    if len(item) > 2 and len(item) < 50:  # Reasonable length for a skill
                        skills.add(item.title())
        
        return list(skills)
    
    def analyze_cv_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the overall structure and quality of the CV"""
        analysis = {
            "language": self.detect_language(text),
            "length": len(text),
            "sections_found": [],
            "completeness_score": 0,
            "quality_indicators": []
        }
        
        # Check for common CV sections
        language = analysis["language"]
        all_keywords = self.mongolian_keywords if language == "mn" else self.english_keywords
        
        sections_score = 0
        for section_type, keywords in all_keywords.items():
            found = any(keyword.lower() in text.lower() for keyword in keywords)
            if found:
                analysis["sections_found"].append(section_type)
                sections_score += 1
        
        # Calculate completeness score
        max_sections = len(all_keywords)
        analysis["completeness_score"] = (sections_score / max_sections) * 100
        
        # Quality indicators
        if analysis["length"] > 1000:
            analysis["quality_indicators"].append("Adequate length")
        if len(analysis["sections_found"]) >= 4:
            analysis["quality_indicators"].append("Well-structured")
        if "contact" in analysis["sections_found"]:
            analysis["quality_indicators"].append("Contact information present")
        
        return analysis
        
    def parse_cv(self, file_path: str) -> ParsedCV:
        """Enhanced CV parsing with bilingual support"""
        try:
            logger.info(f"🔍 Parsing CV: {file_path}")
            
            # Extract raw text from file
            raw_text = extract_text_from_file(file_path)
            if not raw_text:
                raise ValueError(f"Could not extract text from {file_path}")
            
            # Clean the text
            cleaned_text = clean_text(raw_text)
            
            # Detect language
            language = self.detect_language(cleaned_text)
            logger.info(f"📝 Detected language: {'Mongolian' if language == 'mn' else 'English'}")
            
            # Analyze CV structure
            structure_analysis = self.analyze_cv_structure(cleaned_text)
            
            # Extract information using language-specific patterns
            personal_info = self.parse_personal_info(cleaned_text, language)
            education = self.parse_education(cleaned_text, language)
            experience = self.parse_experience(cleaned_text, language)
            skills = self.parse_skills(cleaned_text, language)
            
            # Use LLM for enhanced extraction
            llm_data = self._extract_structured_data_with_llm(cleaned_text, language)
            
            # Extract basic information using regex patterns
            email = personal_info.get('email') or extract_email_from_text(raw_text)
            phone = personal_info.get('phone') or extract_phone_from_text(raw_text)
            experience_years = extract_years_of_experience(raw_text)
            
            # Combine all extracted data
            parsed_cv = ParsedCV(
                name=format_candidate_name(
                    personal_info.get('name') or 
                    llm_data.get('name', 'Unknown Candidate')
                ),
                email=email or llm_data.get('email'),
                phone=phone or llm_data.get('phone'),
                location=personal_info.get('location') or llm_data.get('location'),
                current_role=llm_data.get('current_role'),
                experience_years=experience_years or llm_data.get('experience_years'),
                skills=list(set(skills + llm_data.get('skills', []))),  # Combine and deduplicate
                education=education or llm_data.get('education', []),
                certifications=llm_data.get('certifications', []),
                work_experience=experience or llm_data.get('work_experience', []),
                languages=llm_data.get('languages', []),
                summary=llm_data.get('summary'),
                raw_text=raw_text,
                file_name=file_path.split('/')[-1]
            )
            
            logger.info(f"✅ Successfully parsed CV for {parsed_cv.name}")
            if parsed_cv.skills:
                logger.info(f"   Skills found: {len(parsed_cv.skills)}")
            if parsed_cv.experience_years:
                logger.info(f"   Experience: {parsed_cv.experience_years} years")
            
            return parsed_cv
            
        except Exception as e:
            logger.error(f"❌ Error parsing CV {file_path}: {str(e)}")
            # Return minimal ParsedCV with error info
            return ParsedCV(
                name="Unknown Candidate",
                raw_text=f"Error parsing file: {str(e)}",
                file_name=file_path.split('/')[-1]
            )
    
    def _extract_structured_data_with_llm(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Enhanced LLM extraction with bilingual prompts"""
        
        if language == "mn":
            system_prompt = """Та мэргэжлийн CV шинжлэгч юм. Өгөгдсөн CV текстээс бүтэцтэй мэдээлэл гарган авч JSON объект болгон буцаана уу.

Дараах мэдээллийг гаргана уу:
- name: Ажилтны бүтэн нэр
- email: Имэйл хаяг
- phone: Утасны дугаар
- location: Одоогийн байршил/хаяг
- current_role: Одоогийн эсвэл сүүлийн ажлын албан тушаал
- experience_years: Нийт ажлын туршлагын жил (тоогоор)
- skills: Техникийн болон мэргэжлийн чадваруудын жагсаалт
- education: Боловсролын мэдээлэл (зэрэг, сургууль, он)
- certifications: Мэргэжлийн гэрчилгээний жагсаалт
- work_experience: Ажлын түүх (компани, албан тушаал, үргэлжлэх хугацаа, үүрэг)
- languages: Хэлний чадварын жагсаалт
- summary: Мэргэжлийн товчоон эсвэл зорилго

Зөвхөн JSON объект буцаана уу. Мэдээлэл байхгүй бол null эсвэл хоосон array ашиглана уу."""
            
            human_prompt = f"Дараах CV текстийг шинжлэн бүтэцтэй мэдээлэл гаргана уу:\n\n{text}"
        else:
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

Return ONLY a valid JSON object. If information is not available, use null or empty array as appropriate."""

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
            logger.error(f"Error parsing LLM JSON response: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error with LLM extraction: {str(e)}")
            return {}
    
    def parse_multiple_cvs(self, file_paths: List[str]) -> List[ParsedCV]:
        """Parse multiple CV files with progress tracking"""
        parsed_cvs = []
        
        logger.info(f"🚀 Starting to parse {len(file_paths)} CV files")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"📄 Processing CV {i}/{len(file_paths)}: {file_path}")
            parsed_cv = self.parse_cv(file_path)
            parsed_cvs.append(parsed_cv)
        
        logger.info(f"✅ Completed parsing {len(parsed_cvs)} CVs")
        return parsed_cvs
    
    def process(self, state: AgentState) -> AgentState:
        """Process CV files in the agent state with enhanced error handling"""
        if not state.cv_files:
            state.errors.append("No CV files provided for parsing")
            return state
        
        try:
            logger.info("🔍 CV Analyzer Agent: Starting CV parsing...")
            state.current_step = "parsing_cvs"
            
            # Parse all CV files
            parsed_cvs = self.parse_multiple_cvs(state.cv_files)
            state.parsed_cvs = parsed_cvs
            
            logger.info(f"✅ CV Analyzer Agent: Successfully parsed {len(parsed_cvs)} CVs")
            
            # Log parsing results with language detection
            for cv in parsed_cvs:
                language = self.detect_language(cv.raw_text)
                lang_name = "Mongolian" if language == "mn" else "English"
                logger.info(f"   - {cv.name} ({cv.file_name}) - {lang_name}")
                if cv.skills:
                    logger.info(f"     Skills: {', '.join(cv.skills[:5])}{'...' if len(cv.skills) > 5 else ''}")
                if cv.experience_years:
                    logger.info(f"     Experience: {cv.experience_years} years")
            
            state.current_step = "cvs_parsed"
            
        except Exception as e:
            error_msg = f"CV Analyzer Agent error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state
    
    def _process_response(self, response: str, state: AgentState, context: Dict[str, Any]) -> AgentState:
        """Process LLM response for CV parsing"""
        try:
            # For CV parsing, we use the traditional approach
            # This method is called by the enhanced base agent's process method
            if not state.cv_files:
                state.errors.append("No CV files provided for parsing")
                return state
            
            logger.info("🔍 CV Parser Agent: Starting CV parsing...")
            state.current_step = "parsing_cvs"
            
            # Parse all CV files
            parsed_cvs = self.parse_multiple_cvs(state.cv_files)
            state.parsed_cvs = parsed_cvs
            
            logger.info(f"✅ CV Parser Agent: Successfully parsed {len(parsed_cvs)} CVs")
            
            # Log parsing results with language detection
            for cv in parsed_cvs:
                language = self.detect_language(cv.raw_text)
                lang_name = "Mongolian" if language == "mn" else "English"
                logger.info(f"   - {cv.name} ({cv.file_name}) - {lang_name}")
                if cv.skills:
                    logger.info(f"     Skills: {', '.join(cv.skills[:5])}{'...' if len(cv.skills) > 5 else ''}")
                if cv.experience_years:
                    logger.info(f"     Experience: {cv.experience_years} years")
            
            state.current_step = "cvs_parsed"
            
        except Exception as e:
            error_msg = f"CV Parser Agent error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state

# Backward compatibility alias
CVAnalyzerAgent = CVParserAgent 