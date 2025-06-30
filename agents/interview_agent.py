from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import json

from models import CandidateScore, JobDescription, InterviewQuestion, CandidateQuestions, AgentState
from config import Config

class InterviewAgent:
    """Agent responsible for generating tailored interview questions for shortlisted candidates"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.get_gemini_api_key(),
            temperature=Config.TEMPERATURE
        )
    
    def generate_questions_for_candidate(self, candidate: CandidateScore, 
                                       job_description: JobDescription) -> CandidateQuestions:
        """Generate tailored interview questions for a specific candidate"""
        try:
            # Generate different types of questions
            technical_questions = self._generate_technical_questions(candidate, job_description)
            behavioral_questions = self._generate_behavioral_questions(candidate, job_description)
            role_specific_questions = self._generate_role_specific_questions(candidate, job_description)
            
            candidate_questions = CandidateQuestions(
                candidate_name=candidate.candidate_name,
                job_title=job_description.title,
                technical_questions=technical_questions,
                behavioral_questions=behavioral_questions,
                role_specific_questions=role_specific_questions,
                total_questions=len(technical_questions) + len(behavioral_questions) + len(role_specific_questions)
            )
            
            return candidate_questions
            
        except Exception as e:
            print(f"Error generating questions for {candidate.candidate_name}: {str(e)}")
            # Return minimal questions set
            return CandidateQuestions(
                candidate_name=candidate.candidate_name,
                job_title=job_description.title,
                technical_questions=[],
                behavioral_questions=[],
                role_specific_questions=[],
                total_questions=0
            )
    
    def _generate_technical_questions(self, candidate: CandidateScore, 
                                    job_description: JobDescription) -> List[InterviewQuestion]:
        """Generate technical questions based on candidate's skills and job requirements"""
        
        system_prompt = """You are an expert technical interviewer. Generate technical interview questions based on the candidate's background and job requirements.

Return your response as a JSON array of question objects with this structure:
[
    {
        "question": "The actual interview question",
        "category": "technical",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Key point 1", "Key point 2", "Key point 3"]
    }
]

Generate 3-5 technical questions that:
1. Test the candidate's claimed technical skills
2. Are relevant to the job requirements
3. Vary in difficulty (mix of easy, medium, hard)
4. Focus on practical application rather than just theory
5. Consider the candidate's experience level

If the candidate has gaps in required skills, include questions that explore those areas."""

        # Prepare context
        context = f"""
CANDIDATE PROFILE:
- Name: {candidate.candidate_name}
- Matched Skills: {', '.join(candidate.matched_skills) if candidate.matched_skills else 'None specified'}
- Missing Skills: {', '.join(candidate.missing_skills) if candidate.missing_skills else 'None'}
- Strengths: {', '.join(candidate.strengths) if candidate.strengths else 'None specified'}
- Overall Score: {candidate.overall_score:.1f}/100

JOB REQUIREMENTS:
- Title: {job_description.title}
- Required Skills: {', '.join(job_description.required_skills) if job_description.required_skills else 'None specified'}
- Preferred Skills: {', '.join(job_description.preferred_skills) if job_description.preferred_skills else 'None specified'}
- Min Experience: {job_description.min_experience or 'Not specified'} years

Generate technical interview questions for this candidate."""

        return self._get_questions_from_llm(system_prompt, context, "technical")
    
    def _generate_behavioral_questions(self, candidate: CandidateScore, 
                                     job_description: JobDescription) -> List[InterviewQuestion]:
        """Generate behavioral questions based on candidate's background and role requirements"""
        
        system_prompt = """You are an expert HR interviewer. Generate behavioral interview questions using the STAR method (Situation, Task, Action, Result).

Return your response as a JSON array of question objects with this structure:
[
    {
        "question": "The actual interview question",
        "category": "behavioral",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Key point 1", "Key point 2", "Key point 3"]
    }
]

Generate 3-4 behavioral questions that:
1. Explore the candidate's past experiences and behaviors
2. Are relevant to the job requirements and company culture
3. Use the "Tell me about a time when..." format
4. Focus on key competencies like leadership, problem-solving, teamwork, etc.
5. Consider the candidate's strengths and potential weaknesses"""

        context = f"""
CANDIDATE PROFILE:
- Name: {candidate.candidate_name}
- Strengths: {', '.join(candidate.strengths) if candidate.strengths else 'None specified'}
- Weaknesses: {', '.join(candidate.weaknesses) if candidate.weaknesses else 'None specified'}
- Recommendation: {candidate.recommendation}

JOB REQUIREMENTS:
- Title: {job_description.title}
- Company: {job_description.company}
- Key Responsibilities: {', '.join(job_description.responsibilities) if job_description.responsibilities else 'Not specified'}

Generate behavioral interview questions for this candidate."""

        return self._get_questions_from_llm(system_prompt, context, "behavioral")
    
    def _generate_role_specific_questions(self, candidate: CandidateScore, 
                                        job_description: JobDescription) -> List[InterviewQuestion]:
        """Generate role-specific questions based on the job description and candidate's background"""
        
        system_prompt = """You are an expert interviewer for this specific role. Generate role-specific interview questions that test the candidate's understanding of the role and industry.

Return your response as a JSON array of question objects with this structure:
[
    {
        "question": "The actual interview question",
        "category": "role-specific",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Key point 1", "Key point 2", "Key point 3"]
    }
]

Generate 2-3 role-specific questions that:
1. Test understanding of the specific role and its challenges
2. Explore industry knowledge and trends
3. Assess cultural fit and motivation
4. Are tailored to this specific position and company
5. Help determine if the candidate is genuinely interested in this role"""

        context = f"""
CANDIDATE PROFILE:
- Name: {candidate.candidate_name}
- Overall Score: {candidate.overall_score:.1f}/100
- Recommendation: {candidate.recommendation}

JOB DETAILS:
- Title: {job_description.title}
- Company: {job_description.company}
- Location: {job_description.location or 'Not specified'}
- Job Type: {job_description.job_type or 'Not specified'}
- Description: {job_description.description[:300]}...

Generate role-specific interview questions for this position."""

        return self._get_questions_from_llm(system_prompt, context, "role-specific")
    
    def _get_questions_from_llm(self, system_prompt: str, context: str, category: str) -> List[InterviewQuestion]:
        """Get questions from LLM and parse them"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                questions_data = json.loads(json_text)
            else:
                questions_data = json.loads(response_text)
            
            # Convert to InterviewQuestion objects
            questions = []
            for q_data in questions_data:
                question = InterviewQuestion(
                    question=q_data.get('question', ''),
                    category=q_data.get('category', category),
                    difficulty=q_data.get('difficulty', 'medium'),
                    expected_answer_points=q_data.get('expected_answer_points', [])
                )
                questions.append(question)
            
            return questions
            
        except json.JSONDecodeError as e:
            print(f"Error parsing {category} questions JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"Error generating {category} questions: {str(e)}")
            return []
    
    def generate_questions_for_all_candidates(self, shortlisted_candidates: List[CandidateScore], 
                                           job_description: JobDescription) -> Dict[str, CandidateQuestions]:
        """Generate interview questions for all shortlisted candidates"""
        all_questions = {}
        
        for candidate in shortlisted_candidates:
            print(f"Generating questions for: {candidate.candidate_name}")
            questions = self.generate_questions_for_candidate(candidate, job_description)
            all_questions[candidate.candidate_name] = questions
        
        return all_questions
    
    def process(self, state: AgentState) -> AgentState:
        """Process interview question generation in the agent state"""
        if not state.shortlisted_candidates:
            state.errors.append("No shortlisted candidates available for interview question generation")
            return state
        
        if not state.job_description:
            state.errors.append("No job description available for interview question generation")
            return state
        
        try:
            print("❓ Interview Agent: Generating interview questions...")
            state.current_step = "generating_questions"
            
            # Generate questions for all shortlisted candidates
            interview_questions = self.generate_questions_for_all_candidates(
                state.shortlisted_candidates, state.job_description
            )
            state.interview_questions = interview_questions
            
            print(f"✅ Interview Agent: Generated questions for {len(interview_questions)} candidates")
            
            # Log question generation results
            total_questions = 0
            for candidate_name, questions in interview_questions.items():
                total_questions += questions.total_questions
                print(f"   - {candidate_name}: {questions.total_questions} questions")
                print(f"     Technical: {len(questions.technical_questions)}, "
                      f"Behavioral: {len(questions.behavioral_questions)}, "
                      f"Role-specific: {len(questions.role_specific_questions)}")
            
            print(f"   Total questions generated: {total_questions}")
            
            state.current_step = "questions_generated"
            
        except Exception as e:
            error_msg = f"Interview Agent error: {str(e)}"
            print(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 