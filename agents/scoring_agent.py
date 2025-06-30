from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import json

from models import ParsedCV, JobDescription, CandidateScore, AgentState
from utils import calculate_skill_match_percentage
from config import Config

class ScoringAgent:
    """Agent responsible for scoring candidates against job requirements"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.get_gemini_api_key(),
            temperature=Config.TEMPERATURE
        )
    
    def score_candidate(self, parsed_cv: ParsedCV, job_description: JobDescription) -> CandidateScore:
        """Score a single candidate against the job description"""
        try:
            # Calculate basic skill match
            skills_match_score = self._calculate_skills_score(parsed_cv, job_description)
            
            # Calculate experience score
            experience_score = self._calculate_experience_score(parsed_cv, job_description)
            
            # Calculate education score
            education_score = self._calculate_education_score(parsed_cv, job_description)
            
            # Use LLM for comprehensive analysis
            llm_analysis = self._get_llm_analysis(parsed_cv, job_description)
            
            # Calculate overall score (weighted average)
            overall_score = (
                skills_match_score * 0.4 +  # 40% weight on skills
                experience_score * 0.3 +    # 30% weight on experience
                education_score * 0.2 +     # 20% weight on education
                llm_analysis.get('cultural_fit_score', 70) * 0.1  # 10% weight on cultural fit
            )
            
            # Identify matched and missing skills
            matched_skills = self._get_matched_skills(parsed_cv, job_description)
            missing_skills = self._get_missing_skills(parsed_cv, job_description)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(overall_score, llm_analysis)
            
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
            
            return candidate_score
            
        except Exception as e:
            print(f"Error scoring candidate {parsed_cv.name}: {str(e)}")
            # Return default score with error info
            return CandidateScore(
                candidate_name=parsed_cv.name,
                file_name=parsed_cv.file_name,
                skills_match_score=0,
                experience_score=0,
                education_score=0,
                overall_score=0,
                recommendation="Error in evaluation",
                reasoning=f"Error occurred during scoring: {str(e)}"
            )
    
    def _calculate_skills_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Calculate skills matching score"""
        if not job_description.required_skills:
            return 100.0  # If no specific skills required, give full score
        
        # Calculate match percentage for required skills
        required_match = calculate_skill_match_percentage(
            parsed_cv.skills, job_description.required_skills
        )
        
        # Calculate match percentage for preferred skills (bonus points)
        preferred_match = 0
        if job_description.preferred_skills:
            preferred_match = calculate_skill_match_percentage(
                parsed_cv.skills, job_description.preferred_skills
            ) * 0.2  # 20% bonus for preferred skills
        
        # Combine scores (cap at 100)
        total_score = min(required_match + preferred_match, 100)
        
        return total_score
    
    def _calculate_experience_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Calculate experience score"""
        if not job_description.min_experience:
            return 100.0  # If no experience requirement, give full score
        
        candidate_experience = parsed_cv.experience_years or 0
        required_experience = job_description.min_experience
        
        if candidate_experience >= required_experience:
            # Give full score if meets requirement, bonus for extra experience
            bonus = min((candidate_experience - required_experience) * 5, 20)  # Up to 20% bonus
            return min(100 + bonus, 100)
        else:
            # Penalize for lack of experience
            score = (candidate_experience / required_experience) * 100
            return max(score, 0)
    
    def _calculate_education_score(self, parsed_cv: ParsedCV, job_description: JobDescription) -> float:
        """Calculate education score"""
        if not job_description.education_requirements:
            return 100.0  # If no education requirement, give full score
        
        if not parsed_cv.education:
            return 30.0  # Some score for no education info
        
        # Simple scoring based on education level
        education_text = ' '.join([str(edu) for edu in parsed_cv.education]).lower()
        
        score = 50  # Base score
        
        # Check for degree levels
        if any(keyword in education_text for keyword in ['phd', 'doctorate', 'ph.d']):
            score = 100
        elif any(keyword in education_text for keyword in ['master', 'msc', 'mba', 'ma']):
            score = 90
        elif any(keyword in education_text for keyword in ['bachelor', 'bsc', 'ba', 'degree']):
            score = 80
        elif any(keyword in education_text for keyword in ['diploma', 'certificate']):
            score = 70
        
        return score
    
    def _get_llm_analysis(self, parsed_cv: ParsedCV, job_description: JobDescription) -> Dict[str, Any]:
        """Get comprehensive analysis from LLM"""
        
        system_prompt = """You are an expert HR recruiter. Analyze the candidate's CV against the job description and provide a comprehensive evaluation.

Return your analysis as a JSON object with the following structure:
{
    "cultural_fit_score": 75,
    "strengths": ["List of candidate strengths"],
    "weaknesses": ["List of areas for improvement"],
    "reasoning": "Detailed reasoning for the evaluation",
    "key_highlights": ["Notable achievements or qualifications"],
    "concerns": ["Any concerns about the candidate"]
}

Focus on:
1. How well the candidate's background aligns with the role
2. Relevant achievements and experience
3. Potential cultural fit
4. Areas where the candidate excels
5. Areas where the candidate might need development
6. Overall suitability for the position

Provide honest, constructive feedback that would help in making hiring decisions."""

        # Prepare candidate summary
        cv_summary = f"""
Candidate: {parsed_cv.name}
Current Role: {parsed_cv.current_role or 'Not specified'}
Experience: {parsed_cv.experience_years or 'Not specified'} years
Skills: {', '.join(parsed_cv.skills) if parsed_cv.skills else 'Not specified'}
Education: {parsed_cv.education if parsed_cv.education else 'Not specified'}
Summary: {parsed_cv.summary or 'Not provided'}
"""

        # Prepare job summary
        job_summary = f"""
Job Title: {job_description.title}
Company: {job_description.company}
Required Skills: {', '.join(job_description.required_skills) if job_description.required_skills else 'Not specified'}
Preferred Skills: {', '.join(job_description.preferred_skills) if job_description.preferred_skills else 'Not specified'}
Min Experience: {job_description.min_experience or 'Not specified'} years
Education Requirements: {', '.join(job_description.education_requirements) if job_description.education_requirements else 'Not specified'}
Job Description: {job_description.description[:500]}...
"""

        human_prompt = f"""Analyze this candidate against the job requirements:

CANDIDATE PROFILE:
{cv_summary}

JOB REQUIREMENTS:
{job_summary}

Provide your analysis as a JSON object."""

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
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM analysis JSON: {str(e)}")
            return {
                "cultural_fit_score": 70,
                "strengths": ["Unable to analyze"],
                "weaknesses": ["Analysis unavailable"],
                "reasoning": "LLM analysis failed"
            }
        except Exception as e:
            print(f"Error with LLM analysis: {str(e)}")
            return {
                "cultural_fit_score": 70,
                "strengths": ["Unable to analyze"],
                "weaknesses": ["Analysis unavailable"],
                "reasoning": "LLM analysis failed"
            }
    
    def _get_matched_skills(self, parsed_cv: ParsedCV, job_description: JobDescription) -> List[str]:
        """Get list of skills that match job requirements"""
        if not parsed_cv.skills or not job_description.required_skills:
            return []
        
        candidate_skills_lower = [skill.lower() for skill in parsed_cv.skills]
        required_skills_lower = [skill.lower() for skill in job_description.required_skills]
        
        matched = []
        for req_skill in job_description.required_skills:
            if req_skill.lower() in candidate_skills_lower:
                matched.append(req_skill)
        
        return matched
    
    def _get_missing_skills(self, parsed_cv: ParsedCV, job_description: JobDescription) -> List[str]:
        """Get list of required skills that candidate doesn't have"""
        if not job_description.required_skills:
            return []
        
        candidate_skills_lower = [skill.lower() for skill in parsed_cv.skills] if parsed_cv.skills else []
        
        missing = []
        for req_skill in job_description.required_skills:
            if req_skill.lower() not in candidate_skills_lower:
                missing.append(req_skill)
        
        return missing
    
    def _generate_recommendation(self, overall_score: float, llm_analysis: Dict[str, Any]) -> str:
        """Generate hiring recommendation based on score"""
        if overall_score >= 85:
            return "Highly Recommended - Strong candidate with excellent fit"
        elif overall_score >= 70:
            return "Recommended - Good candidate with solid qualifications"
        elif overall_score >= 60:
            return "Consider - Candidate has potential but may need development"
        elif overall_score >= 40:
            return "Weak Candidate - Significant gaps in requirements"
        else:
            return "Not Recommended - Poor fit for the role"
    
    def score_all_candidates(self, parsed_cvs: List[ParsedCV], job_description: JobDescription) -> List[CandidateScore]:
        """Score all candidates against the job description"""
        candidate_scores = []
        
        for parsed_cv in parsed_cvs:
            print(f"Scoring candidate: {parsed_cv.name}")
            score = self.score_candidate(parsed_cv, job_description)
            candidate_scores.append(score)
        
        # Sort by overall score (descending)
        candidate_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
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
            print("📊 Scoring Agent: Starting candidate evaluation...")
            state.current_step = "scoring_candidates"
            
            # Score all candidates
            candidate_scores = self.score_all_candidates(state.parsed_cvs, state.job_description)
            state.candidate_scores = candidate_scores
            
            print(f"✅ Scoring Agent: Successfully scored {len(candidate_scores)} candidates")
            
            # Log scoring results
            for i, score in enumerate(candidate_scores[:10], 1):  # Show top 10
                print(f"   {i}. {score.candidate_name}: {score.overall_score:.1f}/100 - {score.recommendation}")
            
            state.current_step = "candidates_scored"
            
        except Exception as e:
            error_msg = f"Scoring Agent error: {str(e)}"
            print(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 