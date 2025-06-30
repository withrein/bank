import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Optional, Any
import json

from models import CandidateScore, JobDescription, InterviewQuestion, CandidateQuestions, AgentState
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewAgent:
    """Enhanced Interview Agent with bilingual support for generating tailored interview questions"""
    
    def __init__(self):
        model_config = Config.get_current_model_config()
        self.llm = ChatOpenAI(
            model=model_config["model"],
            openai_api_key=model_config["api_key"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )
        self.mongolian_keywords = Config.get_language_keywords("mn")
        self.english_keywords = Config.get_language_keywords("en")
    
    def detect_language_preference(self, candidate: CandidateScore, job_description: JobDescription) -> str:
        """Detect preferred language for interview questions"""
        # Check job description language
        job_text = f"{job_description.title} {job_description.company} {job_description.description}".lower()
        
        # Count Cyrillic characters
        cyrillic_count = sum(1 for char in job_text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in job_text if char.isalpha() and ord(char) < 256)
        
        total_alpha = cyrillic_count + latin_count
        if total_alpha > 0 and (cyrillic_count / total_alpha) > 0.3:
            return "mn"
        
        # Check for Mongolian keywords in job description
        mongolian_keywords_found = sum(1 for keyword_list in self.mongolian_keywords.values() 
                                     for keyword in keyword_list if keyword in job_text)
        
        return "mn" if mongolian_keywords_found >= 2 else "en"
    
    def generate_questions_for_candidate(self, candidate: CandidateScore, 
                                       job_description: JobDescription) -> CandidateQuestions:
        """Generate tailored interview questions for a specific candidate with bilingual support"""
        try:
            logger.info(f"🎤 Generating interview questions for {candidate.candidate_name}")
            
            # Detect preferred language
            language = self.detect_language_preference(candidate, job_description)
            logger.info(f"📝 Using {'Mongolian' if language == 'mn' else 'English'} for questions")
            
            # Generate different types of questions
            technical_questions = self._generate_technical_questions(candidate, job_description, language)
            behavioral_questions = self._generate_behavioral_questions(candidate, job_description, language)
            role_specific_questions = self._generate_role_specific_questions(candidate, job_description, language)
            
            # Add general questions for cultural fit
            general_questions = self._generate_general_questions(candidate, job_description, language)
            
            total_questions = (len(technical_questions) + len(behavioral_questions) + 
                             len(role_specific_questions) + len(general_questions))
            
            candidate_questions = CandidateQuestions(
                candidate_name=candidate.candidate_name,
                job_title=job_description.title,
                technical_questions=technical_questions,
                behavioral_questions=behavioral_questions,
                role_specific_questions=role_specific_questions,
                total_questions=total_questions
            )
            
            logger.info(f"✅ Generated {total_questions} questions for {candidate.candidate_name}")
            return candidate_questions
            
        except Exception as e:
            logger.error(f"❌ Error generating questions for {candidate.candidate_name}: {str(e)}")
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
                                    job_description: JobDescription, language: str = "en") -> List[InterviewQuestion]:
        """Generate technical questions with bilingual support"""
        
        if language == "mn":
            system_prompt = """Та мэргэжлийн техникийн ярилцлага авдаг мэргэжилтэн юм. Ажилтны мэдлэг чадвар болон ажлын шаардлагад тулгуурлан техникийн асуултууд үүсгэнэ үү.

Дараах JSON хэлбэрээр хариулна уу:
[
    {
        "question": "Бодит ярилцлагын асуулт",
        "category": "technical",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Гол санаа 1", "Гол санаа 2", "Гол санаа 3"]
    }
]

3-5 техникийн асуулт үүсгэнэ үү:
1. Ажилтны техникийн чадварыг шалгах
2. Ажлын шаардлагатай холбоотой
3. Түвшний хувьд олон янз байх (амархан, дунд, хүнд)
4. Онолын биш, практик хэрэглээнд чиглэсэн
5. Ажилтны туршлагын түвшинд тохирсон

Хэрэв ажилтанд шаардлагатай чадвар дутуу байвал тэр талаар асуулт оруулна уу."""
            
            context = f"""
АЖИЛТНЫ МЭДЭЭЛЭЛ:
- Нэр: {candidate.candidate_name}
- Тохирсон чадвар: {', '.join(candidate.matched_skills) if candidate.matched_skills else 'Заагаагүй'}
- Дутуу чадвар: {', '.join(candidate.missing_skills) if candidate.missing_skills else 'Байхгүй'}
- Давуу тал: {', '.join(candidate.strengths) if candidate.strengths else 'Заагаагүй'}
- Нийт оноо: {candidate.overall_score:.1f}/100

АЖЛЫН ШААРДЛАГА:
- Албан тушаал: {job_description.title}
- Шаардлагатай чадвар: {', '.join(job_description.required_skills) if job_description.required_skills else 'Заагаагүй'}
- Хүссэн чадвар: {', '.join(job_description.preferred_skills) if job_description.preferred_skills else 'Заагаагүй'}
- Хамгийн бага туршлага: {job_description.min_experience or 'Заагаагүй'} жил

Энэ ажилтанд техникийн асуулт үүсгэнэ үү."""
        else:
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
6. Include questions about missing skills to assess learning ability

If the candidate has gaps in required skills, include questions that explore those areas."""
            
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
                                     job_description: JobDescription, language: str = "en") -> List[InterviewQuestion]:
        """Generate behavioral questions with bilingual support"""
        
        if language == "mn":
            system_prompt = """Та HR мэргэжилтэн юм. STAR аргыг ашиглан (Нөхцөл байдал, Даалгавар, Үйлдэл, Үр дүн) зан төлөвийн асуултууд үүсгэнэ үү.

Дараах JSON хэлбэрээр хариулна уу:
[
    {
        "question": "Бодит ярилцлагын асуулт",
        "category": "behavioral",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Гол санаа 1", "Гол санаа 2", "Гол санаа 3"]
    }
]

3-4 зан төлөвийн асуулт үүсгэнэ үү:
1. Ажилтны өмнөх туршлага, зан төлөвийг судлах
2. Ажлын шаардлага, компанийн соёлтой холбоотой
3. "Танд ... нөхцөл байдал тохиолдсон тухай ярина уу" хэлбэртэй
4. Удирдлага, асуудал шийдэх, багаар ажиллах чадварт чиглэсэн
5. Ажилтны давуу тал, сул талыг харгалзан"""
            
            context = f"""
АЖИЛТНЫ МЭДЭЭЛЭЛ:
- Нэр: {candidate.candidate_name}
- Давуу тал: {', '.join(candidate.strengths) if candidate.strengths else 'Заагаагүй'}
- Сул тал: {', '.join(candidate.weaknesses) if candidate.weaknesses else 'Заагаагүй'}
- Зөвлөмж: {candidate.recommendation}

АЖЛЫН ШААРДЛАГА:
- Албан тушаал: {job_description.title}
- Компани: {job_description.company}
- Үндсэн үүрэг: {', '.join(job_description.responsibilities) if job_description.responsibilities else 'Заагаагүй'}

Энэ ажилтанд зан төлөвийн асуулт үүсгэнэ үү."""
        else:
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
4. Focus on key competencies like leadership, problem-solving, teamwork, communication
5. Consider the candidate's strengths and potential weaknesses
6. Include questions about handling challenges and conflicts"""

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
                                        job_description: JobDescription, language: str = "en") -> List[InterviewQuestion]:
        """Generate role-specific questions with bilingual support"""
        
        if language == "mn":
            system_prompt = """Та тус ажлын байранд мэргэшсэн ярилцлага авдаг мэргэжилтэн юм. Ажлын тодорхойлолт болон ажилтны мэдлэгт тулгуурлан тухайн албан тушаалд зориулсан асуултууд үүсгэнэ үү.

Дараах JSON хэлбэрээр хариулна уу:
[
    {
        "question": "Бодит ярилцлагын асуулт",
        "category": "role-specific",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Гол санаа 1", "Гол санаа 2", "Гол санаа 3"]
    }
]

2-3 албан тушаалд зориулсан асуулт үүсгэнэ үү:
1. Тухайн ажил, түүний сорилтыг ойлгож байгааг шалгах
2. Салбарын мэдлэг, чиг хандлагыг судлах
3. Соёлын тохирол, сэдэл зорилгыг үнэлэх
4. Тус албан тушаал, компанид тусгайлан зориулсан
5. Ажилтны энэ ажилд жинхэнэ сонирхол байгааг тодорхойлох"""
            
            context = f"""
АЖИЛТНЫ МЭДЭЭЛЭЛ:
- Нэр: {candidate.candidate_name}
- Нийт оноо: {candidate.overall_score:.1f}/100
- Зөвлөмж: {candidate.recommendation}

АЖЛЫН ДЭЛГЭРЭНГҮЙ:
- Албан тушаал: {job_description.title}
- Компани: {job_description.company}
- Байршил: {job_description.location or 'Заагаагүй'}
- Ажлын төрөл: {job_description.job_type or 'Заагаагүй'}
- Тодорхойлолт: {job_description.description[:300]}...

Энэ албан тушаалд зориулсан асуулт үүсгэнэ үү."""
        else:
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
2. Explore industry knowledge and current trends
3. Assess cultural fit and motivation
4. Are tailored to this specific position and company
5. Help determine if the candidate is genuinely interested in this role
6. Explore their vision for the role and potential contributions"""

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
    
    def _generate_general_questions(self, candidate: CandidateScore, 
                                  job_description: JobDescription, language: str = "en") -> List[InterviewQuestion]:
        """Generate general questions for cultural fit and motivation"""
        
        if language == "mn":
            system_prompt = """Та ерөнхий ярилцлага авдаг мэргэжилтэн юм. Соёлын тохирол, сэдэл зорилгыг үнэлэх ерөнхий асуултууд үүсгэнэ үү.

Дараах JSON хэлбэрээр хариулна уу:
[
    {
        "question": "Бодит ярилцлагын асуулт",
        "category": "general",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Гол санаа 1", "Гол санаа 2", "Гол санаа 3"]
    }
]

2-3 ерөнхий асуулт үүсгэнэ үү:
1. Соёлын тохирол үнэлэх
2. Урт хугацааны зорилго судлах
3. Ажлын сэдэл, энерги үнэлэх
4. Багтай хамтран ажиллах чадвар
5. Өөрөө дээшлүүлэх хүсэл эрмэлзэл"""
        else:
            system_prompt = """You are an interviewer focusing on cultural fit and general motivation. Generate general interview questions.

Return your response as a JSON array of question objects with this structure:
[
    {
        "question": "The actual interview question",
        "category": "general",
        "difficulty": "easy|medium|hard",
        "expected_answer_points": ["Key point 1", "Key point 2", "Key point 3"]
    }
]

Generate 2-3 general questions that:
1. Assess cultural fit with the company
2. Explore long-term career goals
3. Evaluate work motivation and energy
4. Test collaboration and communication skills
5. Understand self-improvement mindset"""

        context_base = f"""
CANDIDATE: {candidate.candidate_name}
COMPANY: {job_description.company}
ROLE: {job_description.title}
"""

        return self._get_questions_from_llm(system_prompt, context_base, "general")
    
    def _get_questions_from_llm(self, system_prompt: str, context: str, category: str) -> List[InterviewQuestion]:
        """Get questions from LLM and parse them with improved error handling"""
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
                if isinstance(q_data, dict) and 'question' in q_data:
                    question = InterviewQuestion(
                        question=q_data.get('question', ''),
                        category=q_data.get('category', category),
                        difficulty=q_data.get('difficulty', 'medium'),
                        expected_answer_points=q_data.get('expected_answer_points', [])
                    )
                    questions.append(question)
            
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON response for {category} questions: {str(e)}")
            return self._get_fallback_questions(category)
        except Exception as e:
            logger.error(f"Error with LLM for {category} questions: {str(e)}")
            return self._get_fallback_questions(category)
    
    def _get_fallback_questions(self, category: str) -> List[InterviewQuestion]:
        """Get fallback questions when LLM fails"""
        fallback_questions = {
            "technical": [
                InterviewQuestion(
                    question="Can you walk me through your approach to solving a complex technical problem?",
                    category="technical",
                    difficulty="medium",
                    expected_answer_points=["Problem analysis", "Solution design", "Implementation", "Testing"]
                )
            ],
            "behavioral": [
                InterviewQuestion(
                    question="Tell me about a time when you had to work with a difficult team member.",
                    category="behavioral",
                    difficulty="medium",
                    expected_answer_points=["Situation description", "Actions taken", "Outcome", "Lessons learned"]
                )
            ],
            "role-specific": [
                InterviewQuestion(
                    question="What interests you most about this particular role?",
                    category="role-specific",
                    difficulty="easy",
                    expected_answer_points=["Role understanding", "Personal motivation", "Alignment with skills"]
                )
            ],
            "general": [
                InterviewQuestion(
                    question="Where do you see yourself in 5 years?",
                    category="general",
                    difficulty="easy",
                    expected_answer_points=["Career vision", "Growth mindset", "Alignment with company"]
                )
            ]
        }
        
        return fallback_questions.get(category, [])
    
    def generate_questions_for_all_candidates(self, shortlisted_candidates: List[CandidateScore], 
                                           job_description: JobDescription) -> Dict[str, CandidateQuestions]:
        """Generate interview questions for all shortlisted candidates"""
        all_questions = {}
        
        logger.info(f"🎤 Generating interview questions for {len(shortlisted_candidates)} shortlisted candidates")
        
        for i, candidate in enumerate(shortlisted_candidates, 1):
            logger.info(f"📝 Processing candidate {i}/{len(shortlisted_candidates)}: {candidate.candidate_name}")
            candidate_questions = self.generate_questions_for_candidate(candidate, job_description)
            all_questions[candidate.candidate_name] = candidate_questions
        
        logger.info(f"✅ Completed generating interview questions for all candidates")
        
        # Log summary
        total_questions = sum(q.total_questions for q in all_questions.values())
        logger.info(f"📊 Total questions generated: {total_questions}")
        
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
            logger.info("🎤 Interview Agent: Starting interview question generation...")
            state.current_step = "generating_questions"
            
            # Generate questions for all shortlisted candidates
            interview_questions = self.generate_questions_for_all_candidates(
                state.shortlisted_candidates, 
                state.job_description
            )
            state.interview_questions = interview_questions
            
            logger.info(f"✅ Interview Agent: Successfully generated questions for {len(interview_questions)} candidates")
            
            # Log questions summary
            for candidate_name, questions in interview_questions.items():
                logger.info(f"   - {candidate_name}: {questions.total_questions} questions")
                logger.info(f"     Technical: {len(questions.technical_questions)}, "
                          f"Behavioral: {len(questions.behavioral_questions)}, "
                          f"Role-specific: {len(questions.role_specific_questions)}")
            
            state.current_step = "questions_generated"
            
        except Exception as e:
            error_msg = f"Interview Agent error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 