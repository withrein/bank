import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any
import json

from models import ParsedCV, JobDescription, CandidateScore, AgentState
from utils import calculate_skill_match_percentage
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringAgent:
    """Enhanced Scoring Agent with bilingual support and weighted scoring algorithm"""
    
    def __init__(self):
        model_config = Config.get_current_model_config()
        self.llm = ChatOpenAI(
            model=model_config["model"],
            openai_api_key=model_config["api_key"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )
        self.scoring_weights = Config.SCORING_WEIGHTS
        self.mongolian_keywords = Config.get_language_keywords("mn")
        self.english_keywords = Config.get_language_keywords("en")
    
    def detect_cv_language(self, parsed_cv: ParsedCV) -> str:
        """Detect the primary language of the CV"""
        text = parsed_cv.raw_text.lower()
        
        # Count Cyrillic characters for Mongolian detection
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text if char.isalpha() and ord(char) < 256)
        
        total_alpha = cyrillic_count + latin_count
        if total_alpha > 0 and (cyrillic_count / total_alpha) > 0.3:
            return "mn"
        
        # Check for Mongolian keywords
        mongolian_keywords_found = sum(1 for keyword_list in self.mongolian_keywords.values() 
                                     for keyword in keyword_list if keyword in text)
        
        return "mn" if mongolian_keywords_found >= 3 else "en"
    
    def score_candidate(self, parsed_cv: ParsedCV, job_description: JobDescription) -> CandidateScore:
        """Score a single candidate against the job description with enhanced bilingual analysis"""
        try:
            logger.info(f"📊 Scoring candidate: {parsed_cv.name}")
            
            # Detect CV language for appropriate analysis
            cv_language = self.detect_cv_language(parsed_cv)
            
            # Calculate component scores using weighted algorithm
            skills_match_score = self._calculate_skills_score(parsed_cv, job_description)
            experience_score = self._calculate_experience_score(parsed_cv, job_description)
            education_score = self._calculate_education_score(parsed_cv, job_description)
            
            # Use LLM for comprehensive analysis with bilingual support
            llm_analysis = self._get_llm_analysis(parsed_cv, job_description, cv_language)
            
            # Calculate overall score using configured weights
            overall_score = (
                skills_match_score * self.scoring_weights["skills"] +
                experience_score * self.scoring_weights["experience"] +
                education_score * self.scoring_weights["education"] +
                llm_analysis.get('cultural_fit_score', 70) * self.scoring_weights["other"]
            )
            
            # Cap the score at 100
            overall_score = min(overall_score, 100)
            
            # Identify matched and missing skills
            matched_skills = self._get_matched_skills(parsed_cv, job_description)
            missing_skills = self._get_missing_skills(parsed_cv, job_description)
            
            # Generate recommendation in appropriate language
            recommendation = self._generate_recommendation(overall_score, llm_analysis, cv_language)
            
            candidate_score = CandidateScore(
                candidate_name=parsed_cv.name,
                file_name=parsed_cv.file_name,
                skills_match_score=skills_match_score,
                experience_score=experience_score,
                education_score=education_score,
                overall_score=overall_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                strengths=llm_analysis.get('strengths', []),
                weaknesses=llm_analysis.get('weaknesses', []),
                recommendation=recommendation,
                reasoning=llm_analysis.get('reasoning', f"Overall score: {overall_score:.1f}/100")
            )
            
            logger.info(f"✅ Scored {parsed_cv.name}: {overall_score:.1f}/100")
            
            return candidate_score
            
        except Exception as e:
            logger.error(f"❌ Error scoring candidate {parsed_cv.name}: {str(e)}")
            # Return default score with error info
            return CandidateScore(
                candidate_name=parsed_cv.name,
                file_name=parsed_cv.file_name,
                skills_match_score=0,
                experience_score=0,
                education_score=0,
                overall_score=0,
                recommendation="Үнэлгээнд алдаа гарлаа" if self.detect_cv_language(parsed_cv) == "mn" else "Error in evaluation",
                reasoning=f"Error occurred during scoring: {str(e)}"
            )
    
    def _calculate_skills_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Enhanced skills matching with technical and soft skills analysis"""
        if not job_description.required_skills:
            return 100.0
        
        # Calculate match percentage for required skills
        required_match = calculate_skill_match_percentage(
            parsed_cv.skills, job_description.required_skills
        )
        
        # Calculate match percentage for preferred skills (bonus points)
        preferred_match = 0
        if job_description.preferred_skills:
            preferred_match = calculate_skill_match_percentage(
                parsed_cv.skills, job_description.preferred_skills
            ) * 0.3  # 30% bonus for preferred skills
        
        # Check for language skills relevance
        language_bonus = 0
        if parsed_cv.languages:
            # Bonus for multilingual candidates
            if len(parsed_cv.languages) >= 2:
                language_bonus = 5  # 5% bonus for bilingual+
            
            # Additional bonus for English in Mongolian CVs or vice versa
            cv_language = self.detect_cv_language(parsed_cv)
            if cv_language == "mn" and any("english" in lang.lower() for lang in parsed_cv.languages):
                language_bonus += 5
            elif cv_language == "en" and any("mongolian" in lang.lower() for lang in parsed_cv.languages):
                language_bonus += 5
        
        # Combine scores (cap at 100)
        total_score = min(required_match + preferred_match + language_bonus, 100)
        
        return total_score
    
    def _calculate_experience_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Enhanced experience calculation with role relevance analysis"""
        if not job_description.min_experience:
            return 100.0
        
        candidate_experience = parsed_cv.experience_years or 0
        required_experience = job_description.min_experience
        
        # Base score calculation
        if candidate_experience >= required_experience:
            # Full score if meets requirement, diminishing returns for extra experience
            extra_years = candidate_experience - required_experience
            bonus = min(extra_years * 2, 20)  # Up to 20% bonus, 2% per extra year
            base_score = 100
        else:
            # Graduated penalty for insufficient experience
            ratio = candidate_experience / required_experience
            if ratio >= 0.8:  # 80%+ of required experience
                base_score = 80 + (ratio - 0.8) * 100  # 80-100 score range
            elif ratio >= 0.5:  # 50%+ of required experience
                base_score = 50 + (ratio - 0.5) * 100  # 50-80 score range
            else:
                base_score = ratio * 100  # 0-50 score range
            bonus = 0
        
        # Analyze work experience relevance
        relevance_bonus = 0
        if parsed_cv.work_experience:
            job_title_lower = job_description.title.lower()
            for work in parsed_cv.work_experience:
                if isinstance(work, dict):
                    role = work.get('role', '').lower()
                    company = work.get('company', '').lower()
                    
                    # Check for relevant role keywords
                    if any(keyword in role for keyword in job_title_lower.split()):
                        relevance_bonus += 5  # 5% bonus per relevant role
                    
                    # Check for industry relevance (basic check)
                    if 'tech' in company or 'software' in company or 'IT' in company:
                        if 'engineer' in job_title_lower or 'developer' in job_title_lower:
                            relevance_bonus += 3
        
        total_score = min(base_score + bonus + relevance_bonus, 100)
        return max(total_score, 0)
    
    def _calculate_education_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Enhanced education scoring with field relevance"""
        if not job_description.education_requirements:
            return 100.0
        
        if not parsed_cv.education:
            return 40.0  # Base score for missing education info
        
        # Analyze education level and relevance
        education_text = ' '.join([str(edu) for edu in parsed_cv.education]).lower()
        
        # Base score by education level
        base_score = 50
        if any(keyword in education_text for keyword in ['phd', 'doctorate', 'ph.d', 'доктор']):
            base_score = 100
        elif any(keyword in education_text for keyword in ['master', 'msc', 'mba', 'ma', 'магистр']):
            base_score = 95
        elif any(keyword in education_text for keyword in ['bachelor', 'bsc', 'ba', 'degree', 'бакалавр', 'диплом']):
            base_score = 85
        elif any(keyword in education_text for keyword in ['diploma', 'certificate', 'гэрчилгээ']):
            base_score = 75
        elif any(keyword in education_text for keyword in ['college', 'коллеж']):
            base_score = 70
        
        # Field relevance bonus
        relevance_bonus = 0
        job_desc_lower = job_description.description.lower()
        
        # Check for field-specific keywords
        if 'computer' in education_text or 'software' in education_text or 'IT' in education_text:
            if any(keyword in job_desc_lower for keyword in ['software', 'engineer', 'developer', 'tech']):
                relevance_bonus += 10
        
        if 'business' in education_text or 'management' in education_text or 'MBA' in education_text:
            if any(keyword in job_desc_lower for keyword in ['manager', 'business', 'strategy']):
                relevance_bonus += 10
        
        # Check for STEM fields
        if any(keyword in education_text for keyword in ['engineer', 'science', 'mathematics', 'physics']):
            if 'engineer' in job_desc_lower or 'technical' in job_desc_lower:
                relevance_bonus += 8
        
        total_score = min(base_score + relevance_bonus, 100)
        return total_score
    
    def _get_llm_analysis(self, parsed_cv: ParsedCV, job_description: JobDescription, cv_language: str = "en") -> Dict[str, Any]:
        """Enhanced LLM analysis with bilingual support"""
        
        if cv_language == "mn":
            system_prompt = """Та мэргэжлийн HR рекрутер юм. Ажилтны CV-г ажлын байрны тодорхойлолттой харьцуулан дүгнэлт өгч, JSON объект хэлбэрээр буцаана уу.

Дараах бүтэцтэй дүн шинжилгээ хийнэ үү:
{
    "cultural_fit_score": 75,
    "strengths": ["Ажилтны давуу талуудын жагсаалт"],
    "weaknesses": ["Сайжруулах шаардлагатай талууд"],
    "reasoning": "Үнэлгээний дэлгэрэнгүй үндэслэл",
    "key_highlights": ["Онцлох ур чадвар, амжилт"],
    "concerns": ["Анхаарал татаж буй асуудлууд"],
    "language_proficiency": "Хэлний чадварын үнэлгээ",
    "growth_potential": "Хөгжлийн боломж"
}

Дараах зүйлд анхаарал хандуулна уу:
1. Ажилтны мэдлэг туршлага албан тушаалтай хэр нийцэж байгаа
2. Холбогдох амжилт, туршлага
3. Компанийн соёлтой нийцэх боломж
4. Ажилтны давуу талууд
5. Хөгжүүлэх шаардлагатай талууд
6. Ерөнхий тохирол

Шударга, бодитой дүгнэлт өгөөрэй."""
            
            human_prompt = f"""Дараах ажилтны мэдээллийг ажлын байрны шаардлагатай харьцуулан дүгнэнэ үү:

АЖИЛТНЫ МЭДЭЭЛЭЛ:
Нэр: {parsed_cv.name}
Одоогийн албан тушаал: {parsed_cv.current_role or 'Заагаагүй'}
Ажлын туршлага: {parsed_cv.experience_years or 'Заагаагүй'} жил
Чадвар: {', '.join(parsed_cv.skills) if parsed_cv.skills else 'Заагаагүй'}
Боловсрол: {parsed_cv.education if parsed_cv.education else 'Заагаагүй'}
Хэл: {', '.join(parsed_cv.languages) if parsed_cv.languages else 'Заагаагүй'}

АЖЛЫН БАЙРНЫ ШААРДЛАГА:
Албан тушаал: {job_description.title}
Компани: {job_description.company}
Шаардлагатай чадвар: {', '.join(job_description.required_skills) if job_description.required_skills else 'Заагаагүй'}
Хүссэн чадвар: {', '.join(job_description.preferred_skills) if job_description.preferred_skills else 'Заагаагүй'}
Хамгийн бага туршлага: {job_description.min_experience or 'Заагаагүй'} жил
Боловсролын шаардлага: {', '.join(job_description.education_requirements) if job_description.education_requirements else 'Заагаагүй'}
Ажлын тодорхойлолт: {job_description.description[:500]}...

JSON объект хэлбэрээр дүн шинжилгээ өгнө үү."""
        else:
            system_prompt = """You are an expert HR recruiter. Analyze the candidate's CV against the job description and provide a comprehensive evaluation.

Return your analysis as a JSON object with the following structure:
{
    "cultural_fit_score": 75,
    "strengths": ["List of candidate strengths"],
    "weaknesses": ["List of areas for improvement"],
    "reasoning": "Detailed reasoning for the evaluation",
    "key_highlights": ["Notable achievements or qualifications"],
    "concerns": ["Any concerns about the candidate"],
    "language_proficiency": "Assessment of language skills",
    "growth_potential": "Potential for professional growth"
}

Focus on:
1. How well the candidate's background aligns with the role
2. Relevant achievements and experience
3. Potential cultural fit
4. Areas where the candidate excels
5. Areas where the candidate might need development
6. Overall suitability for the position
7. Communication and language capabilities
8. Leadership and growth potential

Provide honest, constructive feedback that would help in making hiring decisions."""

            human_prompt = f"""Analyze this candidate against the job requirements:

CANDIDATE PROFILE:
Name: {parsed_cv.name}
Current Role: {parsed_cv.current_role or 'Not specified'}
Experience: {parsed_cv.experience_years or 'Not specified'} years
Skills: {', '.join(parsed_cv.skills) if parsed_cv.skills else 'Not specified'}
Education: {parsed_cv.education if parsed_cv.education else 'Not specified'}
Languages: {', '.join(parsed_cv.languages) if parsed_cv.languages else 'Not specified'}
Summary: {parsed_cv.summary or 'Not provided'}

JOB REQUIREMENTS:
Job Title: {job_description.title}
Company: {job_description.company}
Required Skills: {', '.join(job_description.required_skills) if job_description.required_skills else 'Not specified'}
Preferred Skills: {', '.join(job_description.preferred_skills) if job_description.preferred_skills else 'Not specified'}
Min Experience: {job_description.min_experience or 'Not specified'} years
Education Requirements: {', '.join(job_description.education_requirements) if job_description.education_requirements else 'Not specified'}
Job Description: {job_description.description[:500]}...

Provide your analysis as a JSON object."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                analysis = json.loads(json_text)
                
                # Ensure cultural_fit_score is within valid range
                if 'cultural_fit_score' in analysis:
                    analysis['cultural_fit_score'] = max(0, min(100, analysis['cultural_fit_score']))
                
                return analysis
            else:
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON response: {str(e)}")
            # Return default analysis
            return {
                "cultural_fit_score": 70,
                "strengths": ["Unable to analyze"],
                "weaknesses": ["Analysis unavailable"],
                "reasoning": "Error in LLM analysis",
                "key_highlights": [],
                "concerns": ["Analysis failed"]
            }
        except Exception as e:
            logger.error(f"Error with LLM analysis: {str(e)}")
            return {
                "cultural_fit_score": 70,
                "strengths": [],
                "weaknesses": [],
                "reasoning": f"Error in analysis: {str(e)}",
                "key_highlights": [],
                "concerns": []
            }
    
    def _get_matched_skills(self, parsed_cv: ParsedCV, job_description: JobDescription) -> List[str]:
        """Get list of matched skills between CV and job requirements"""
        if not job_description.required_skills or not parsed_cv.skills:
            return []
        
        candidate_skills_lower = [skill.lower() for skill in parsed_cv.skills]
        required_skills_lower = [skill.lower() for skill in job_description.required_skills]
        preferred_skills_lower = [skill.lower() for skill in job_description.preferred_skills] if job_description.preferred_skills else []
        
        matched = []
        for skill in parsed_cv.skills:
            skill_lower = skill.lower()
            if skill_lower in required_skills_lower or skill_lower in preferred_skills_lower:
                matched.append(skill)
            else:
                # Check for partial matches
                for req_skill in job_description.required_skills + (job_description.preferred_skills or []):
                    if req_skill.lower() in skill_lower or skill_lower in req_skill.lower():
                        matched.append(skill)
                        break
        
        return list(set(matched))  # Remove duplicates
    
    def _get_missing_skills(self, parsed_cv: ParsedCV, job_description: JobDescription) -> List[str]:
        """Get list of missing required skills"""
        if not job_description.required_skills:
            return []
        
        candidate_skills_lower = [skill.lower() for skill in parsed_cv.skills] if parsed_cv.skills else []
        missing = []
        
        for req_skill in job_description.required_skills:
            req_skill_lower = req_skill.lower()
            
            # Check for exact or partial match
            found = False
            for candidate_skill in candidate_skills_lower:
                if req_skill_lower in candidate_skill or candidate_skill in req_skill_lower:
                    found = True
                    break
            
            if not found:
                missing.append(req_skill)
        
        return missing
    
    def _generate_recommendation(self, overall_score: float, llm_analysis: Dict[str, Any], cv_language: str = "en") -> str:
        """Generate hiring recommendation based on score and analysis"""
        
        if cv_language == "mn":
            if overall_score >= 85:
                recommendation = "Маш сайн зөвлөмж - Үндсэн шаардлагыг бүрэн хангасан тохиромжтой ажилтан"
            elif overall_score >= 75:
                recommendation = "Сайн зөвлөмж - Ихэнх шаардлагыг хангасан ажилтан"
            elif overall_score >= 65:
                recommendation = "Дунд зэрэг зөвлөмж - Зарим талаараа хангалттай"
            elif overall_score >= 50:
                recommendation = "Анхаарлаар - Нэмэлт үнэлгээ шаардлагатай"
            else:
                recommendation = "Зөвлөхгүй - Үндсэн шаардлагыг хангаагүй"
        else:
            if overall_score >= 85:
                recommendation = "Highly Recommended - Excellent fit with strong qualifications"
            elif overall_score >= 75:
                recommendation = "Recommended - Good fit with most requirements met"
            elif overall_score >= 65:
                recommendation = "Consider - Meets basic requirements with some gaps"
            elif overall_score >= 50:
                recommendation = "Caution - Significant gaps in requirements"
            else:
                recommendation = "Not Recommended - Does not meet minimum requirements"
        
        # Add insights from LLM analysis if available
        if llm_analysis.get('key_highlights'):
            if cv_language == "mn":
                recommendation += f" | Давуу тал: {', '.join(llm_analysis['key_highlights'][:2])}"
            else:
                recommendation += f" | Highlights: {', '.join(llm_analysis['key_highlights'][:2])}"
        
        return recommendation
    
    def score_all_candidates(self, parsed_cvs: List[ParsedCV], job_description: JobDescription) -> List[CandidateScore]:
        """Score all candidates and return sorted results"""
        candidate_scores = []
        
        logger.info(f"🎯 Starting to score {len(parsed_cvs)} candidates for {job_description.title}")
        
        for i, parsed_cv in enumerate(parsed_cvs, 1):
            logger.info(f"📊 Scoring candidate {i}/{len(parsed_cvs)}: {parsed_cv.name}")
            score = self.score_candidate(parsed_cv, job_description)
            candidate_scores.append(score)
        
        # Sort by overall score (descending)
        candidate_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info(f"✅ Completed scoring all candidates")
        logger.info(f"📈 Score range: {candidate_scores[-1].overall_score:.1f} - {candidate_scores[0].overall_score:.1f}")
        
        return candidate_scores
    
    def process(self, state: AgentState) -> AgentState:
        """Process candidate scoring in the agent state"""
        if not state.parsed_cvs:
            state.errors.append("No parsed CVs available for scoring")
            return state
        
        if not state.job_description:
            state.errors.append("No job description available for scoring")
            return state
        
        try:
            logger.info("🎯 Scoring Agent: Starting candidate evaluation...")
            state.current_step = "scoring_candidates"
            
            # Score all candidates
            candidate_scores = self.score_all_candidates(state.parsed_cvs, state.job_description)
            state.candidate_scores = candidate_scores
            
            logger.info(f"✅ Scoring Agent: Successfully evaluated {len(candidate_scores)} candidates")
            
            # Log top candidates
            for i, score in enumerate(candidate_scores[:5], 1):
                logger.info(f"   {i}. {score.candidate_name}: {score.overall_score:.1f}/100 - {score.recommendation}")
            
            state.current_step = "candidates_scored"
            
        except Exception as e:
            error_msg = f"Scoring Agent error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 