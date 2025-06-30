#!/usr/bin/env python3
"""
Enhanced Base Agent Class with Advanced Prompt Engineering and Context Management
"""

import logging
from typing import Dict, Any, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from config import Config
from models import AgentState
import json
import time

logger = logging.getLogger(__name__)

class EnhancedBaseAgent:
    """
    Enhanced base agent with sophisticated prompt engineering and context management
    """
    
    def __init__(self, agent_type: str, specialized_instructions: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        self.context_manager = ContextManager()
        self.prompt_engineer = PromptEngineer(agent_type, specialized_instructions)
        self.interaction_history = []
        
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize LLM with enhanced configuration"""
        model_config = Config.get_current_model_config()
        return ChatOpenAI(
            model=model_config["model"],
            openai_api_key=model_config["api_key"],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            model_kwargs={
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        )
    
    def _initialize_memory(self) -> ConversationBufferWindowMemory:
        """Initialize conversation memory"""
        return ConversationBufferWindowMemory(
            k=Config.MODEL_CONTEXT["agent_memory"]["short_term"],
            return_messages=True
        )
    
    def process(self, state: AgentState) -> AgentState:
        """Enhanced processing with context management and error handling"""
        try:
            # Log agent activation
            logger.info(f"🤖 {self.agent_type.upper()} Agent activated")
            
            # Prepare context
            context = self.context_manager.prepare_context(state, self.agent_type)
            
            # Generate enhanced prompt
            system_prompt = self.prompt_engineer.generate_system_prompt(context)
            user_prompt = self.prompt_engineer.generate_user_prompt(state, context)
            
            # Execute LLM call with retry mechanism
            response = self._execute_llm_call(system_prompt, user_prompt)
            
            # Process response and update state
            processed_state = self._process_response(response, state, context)
            
            # Update memory and interaction history
            self._update_memory(user_prompt, response)
            self._log_interaction(context, response)
            
            return processed_state
            
        except Exception as e:
            logger.error(f"❌ {self.agent_type} Agent failed: {str(e)}")
            self._handle_error(state, str(e))
            return state
    
    def _execute_llm_call(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Execute LLM call with retry mechanism"""
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                return response.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"⚠️ LLM call failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _process_response(self, response: str, state: AgentState, context: Dict) -> AgentState:
        """Override this method in child classes for specific processing"""
        raise NotImplementedError("Child classes must implement _process_response")
    
    def _update_memory(self, user_input: str, ai_response: str):
        """Update conversation memory"""
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)
    
    def _log_interaction(self, context: Dict, response: str):
        """Log interaction for analysis and debugging"""
        interaction = {
            "timestamp": time.time(),
            "agent_type": self.agent_type,
            "context_summary": context.get("summary", ""),
            "response_length": len(response),
            "success": True
        }
        self.interaction_history.append(interaction)
    
    def _handle_error(self, state: AgentState, error_msg: str):
        """Handle errors gracefully"""
        if hasattr(state, 'errors'):
            state.errors.append(f"{self.agent_type}: {error_msg}")
        elif isinstance(state, dict):
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_type}: {error_msg}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        successful_interactions = sum(1 for i in self.interaction_history if i["success"])
        avg_response_length = sum(i["response_length"] for i in self.interaction_history) / len(self.interaction_history)
        
        return {
            "total_interactions": len(self.interaction_history),
            "success_rate": successful_interactions / len(self.interaction_history),
            "avg_response_length": avg_response_length,
            "last_interaction": self.interaction_history[-1]["timestamp"]
        }

class ContextManager:
    """Advanced context management for multi-agent workflows"""
    
    def __init__(self):
        self.global_context = {}
        self.agent_contexts = {}
    
    def prepare_context(self, state: AgentState, agent_type: str) -> Dict[str, Any]:
        """Prepare comprehensive context for agent processing"""
        context = {
            "agent_type": agent_type,
            "job_requirements": self._extract_job_context(state),
            "processed_data": self._extract_processed_data(state),
            "workflow_stage": self._determine_workflow_stage(state),
            "previous_results": self._get_previous_results(state),
            "constraints": self._get_constraints(agent_type),
            "summary": ""
        }
        
        # Generate context summary
        context["summary"] = self._generate_context_summary(context)
        
        # Store context for cross-agent communication
        self.agent_contexts[agent_type] = context
        
        return context
    
    def _extract_job_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract job-related context"""
        job_desc = getattr(state, 'job_description', None) or state.get('job_description') if isinstance(state, dict) else None
        if not job_desc:
            return {}
        
        return {
            "title": getattr(job_desc, 'title', '') if hasattr(job_desc, 'title') else job_desc.get('title', ''),
            "required_skills": getattr(job_desc, 'required_skills', []) if hasattr(job_desc, 'required_skills') else job_desc.get('required_skills', []),
            "experience_level": getattr(job_desc, 'min_experience', 0) if hasattr(job_desc, 'min_experience') else job_desc.get('min_experience', 0),
            "industry": getattr(job_desc, 'company', '') if hasattr(job_desc, 'company') else job_desc.get('company', ''),
        }
    
    def _extract_processed_data(self, state: AgentState) -> Dict[str, Any]:
        """Extract already processed data"""
        return {
            "parsed_cvs_count": len(getattr(state, 'parsed_cvs', []) if hasattr(state, 'parsed_cvs') else state.get('parsed_cvs', []) if isinstance(state, dict) else []),
            "scored_candidates_count": len(getattr(state, 'candidate_scores', []) if hasattr(state, 'candidate_scores') else state.get('candidate_scores', []) if isinstance(state, dict) else []),
            "shortlisted_count": len(getattr(state, 'shortlisted_candidates', []) if hasattr(state, 'shortlisted_candidates') else state.get('shortlisted_candidates', []) if isinstance(state, dict) else []),
        }
    
    def _determine_workflow_stage(self, state: AgentState) -> str:
        """Determine current workflow stage"""
        current_step = getattr(state, 'current_step', '') if hasattr(state, 'current_step') else state.get('current_step', '') if isinstance(state, dict) else ''
        if not current_step:
            return "initialization"
        return current_step
    
    def _get_previous_results(self, state: AgentState) -> Dict[str, Any]:
        """Get results from previous agents"""
        return {
            "parsed_cvs": bool(getattr(state, 'parsed_cvs', []) if hasattr(state, 'parsed_cvs') else state.get('parsed_cvs', []) if isinstance(state, dict) else []),
            "candidate_scores": bool(getattr(state, 'candidate_scores', []) if hasattr(state, 'candidate_scores') else state.get('candidate_scores', []) if isinstance(state, dict) else []),
            "shortlisted": bool(getattr(state, 'shortlisted_candidates', []) if hasattr(state, 'shortlisted_candidates') else state.get('shortlisted_candidates', []) if isinstance(state, dict) else []),
        }
    
    def _get_constraints(self, agent_type: str) -> Dict[str, Any]:
        """Get agent-specific constraints"""
        constraints_map = {
            "cv_parser": {
                "max_cvs": 50,
                "supported_formats": Config.ALLOWED_CV_FORMATS,
                "required_fields": ["name", "skills", "experience"]
            },
            "scoring": {
                "score_range": [0, 100],
                "required_criteria": ["skills", "experience", "education"],
                "max_candidates": Config.MAX_CANDIDATES_TO_SHORTLIST * 3
            },
            "shortlisting": {
                "max_shortlisted": Config.MAX_CANDIDATES_TO_SHORTLIST,
                "min_score": Config.MINIMUM_SCORE_THRESHOLD
            },
            "interview": {
                "questions_per_candidate": 15,
                "question_categories": ["technical", "behavioral", "role_specific"]
            },
            "email": {
                "max_emails": 100,
                "supported_types": ["invitation", "rejection", "acknowledgment"]
            }
        }
        return constraints_map.get(agent_type, {})
    
    def _generate_context_summary(self, context: Dict[str, Any]) -> str:
        """Generate a concise context summary"""
        job_context = context.get("job_requirements", {})
        processed = context.get("processed_data", {})
        stage = context.get("workflow_stage", "unknown")
        
        summary = f"Processing {job_context.get('title', 'Unknown Position')} role "
        summary += f"at stage '{stage}'. "
        summary += f"Processed {processed.get('parsed_cvs_count', 0)} CVs, "
        summary += f"scored {processed.get('scored_candidates_count', 0)} candidates, "
        summary += f"shortlisted {processed.get('shortlisted_count', 0)}."
        
        return summary

class PromptEngineer:
    """Advanced prompt engineering for different agent types"""
    
    def __init__(self, agent_type: str, specialized_instructions: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.specialized_instructions = specialized_instructions or {}
        self.base_prompts = Config.PROMPT_ENGINEERING
    
    def generate_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate sophisticated system prompt"""
        base_context = self.base_prompts["system_context"]
        agent_instructions = self.base_prompts["agent_instructions"].get(self.agent_type, {})
        
        system_prompt = f"""
{base_context["role"]}

**EXPERTISE AREAS:**
{chr(10).join(f'• {expertise}' for expertise in base_context["expertise"])}

**INPUT LANGUAGES:** Can process {", ".join(base_context["input_languages"])}
**OUTPUT LANGUAGE:** {base_context["output_language"]} ONLY
**COMMUNICATION TONE:** {base_context["tone"]}

**CRITICAL INSTRUCTION:** {base_context["instruction"]}

**SPECIALIZED INSTRUCTIONS FOR {self.agent_type.upper()} AGENT:**
{self._format_agent_instructions(agent_instructions)}

**CURRENT CONTEXT:**
{context.get("summary", "No context available")}

**WORKFLOW CONSTRAINTS:**
{self._format_constraints(context.get("constraints", {}))}

**QUALITY STANDARDS:**
• Provide accurate, detailed, and actionable outputs
• Maintain consistency with previous agent results
• Consider cultural and linguistic nuances for Mongolian context
• Ensure professional standards in all communications
• Include confidence scores and reasoning for key decisions
• ALWAYS respond in English regardless of input language

**OUTPUT REQUIREMENTS:**
• Use structured JSON format where applicable
• Include metadata and confidence indicators
• Provide clear reasoning for decisions
• ALL OUTPUT MUST BE IN ENGLISH
• Follow established data schemas
"""
        
        return system_prompt.strip()
    
    def generate_user_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate context-aware user prompt"""
        return self._get_agent_specific_prompt(state, context)
    
    def _format_agent_instructions(self, instructions: Dict[str, Any]) -> str:
        """Format agent-specific instructions"""
        if not instructions:
            return "No specific instructions available."
        
        formatted = []
        for key, value in instructions.items():
            if isinstance(value, list):
                formatted.append(f"• {key.replace('_', ' ').title()}: {', '.join(value)}")
            else:
                formatted.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for the agent"""
        if not constraints:
            return "No specific constraints."
        
        formatted = []
        for key, value in constraints.items():
            formatted.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _get_agent_specific_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Get agent-specific prompt based on agent type"""
        prompt_map = {
            "cv_parser": self._generate_cv_parser_prompt,
            "scoring": self._generate_scoring_prompt,
            "shortlisting": self._generate_shortlisting_prompt,
            "interview": self._generate_interview_prompt,
            "email": self._generate_email_prompt
        }
        
        generator = prompt_map.get(self.agent_type, self._generate_generic_prompt)
        return generator(state, context)
    
    def _generate_cv_parser_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate CV parser specific prompt"""
        cv_files = getattr(state, 'cv_files', []) or state.get('cv_files', [])
        job_req = context.get("job_requirements", {})
        
        return f"""
**TASK: CV PARSING AND EXTRACTION**

You need to parse {len(cv_files)} CV files and extract structured information for the position: {job_req.get('title', 'Unknown Position')}.

**EXTRACTION REQUIREMENTS:**
• Candidate name and contact information
• Professional experience with years and roles
• Technical and soft skills
• Educational background
• Certifications and achievements
• Language proficiencies

**SPECIAL ATTENTION:**
• Look for skills matching: {', '.join(job_req.get('required_skills', []))}
• Process both English and Mongolian language CVs
• Extract confidence scores for each field
• Identify potential red flags or inconsistencies
• Translate any Mongolian content to English in your output

**OUTPUT FORMAT:**
Provide structured JSON for each candidate with confidence scores and extracted metadata.
**IMPORTANT: All text in the output must be in English, including candidate information extracted from Mongolian CVs.**
"""
    
    def _generate_scoring_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate scoring agent specific prompt"""
        candidates = getattr(state, 'parsed_cvs', []) or state.get('parsed_cvs', [])
        job_req = context.get("job_requirements", {})
        
        return f"""
**TASK: CANDIDATE SCORING AND EVALUATION**

Evaluate {len(candidates)} candidates for the {job_req.get('title', 'Unknown Position')} position.

**SCORING METHODOLOGY:**
• Skills Match (25%): Alignment with required technical skills
• Experience Relevance (40%): Quality and relevance of work experience
• Education Fit (25%): Educational background alignment
• Cultural Alignment (10%): Soft skills and cultural fit indicators

**EVALUATION CRITERIA:**
• Required Skills: {', '.join(job_req.get('required_skills', []))}
• Experience Level: {job_req.get('experience_level', 'Not specified')} years minimum
• Industry: {job_req.get('industry', 'Not specified')}

**OUTPUT REQUIREMENTS:**
• Overall score (0-100) with detailed breakdown
• Strengths and improvement areas
• Specific skill gaps and matches
• Hiring recommendation with reasoning
• Confidence level for the assessment
**IMPORTANT: All analysis, recommendations, and text must be in English.**
"""
    
    def _generate_shortlisting_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate shortlisting agent specific prompt"""
        scored_candidates = getattr(state, 'candidate_scores', []) or state.get('candidate_scores', [])
        constraints = context.get("constraints", {})
        
        return f"""
**TASK: CANDIDATE SHORTLISTING**

Select the top {constraints.get('max_shortlisted', 5)} candidates from {len(scored_candidates)} evaluated candidates.

**SELECTION CRITERIA:**
• Minimum score threshold: {constraints.get('min_score', 60)}
• Diversity in skill sets and backgrounds
• Potential for growth and cultural fit
• Risk assessment and reliability indicators

**SHORTLISTING STRATEGY:**
• Rank by overall score with additional qualitative factors
• Consider skill complementarity among shortlisted candidates
• Balance experience levels and specializations
• Include reasoning for selection and rejection decisions

**OUTPUT FORMAT:**
• Ranked list of shortlisted candidates
• Rejection summary for non-shortlisted candidates
• Diversity and balance analysis
• Next steps recommendations
**IMPORTANT: All output must be in English.**
"""
    
    def _generate_interview_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate interview agent specific prompt"""
        shortlisted = getattr(state, 'shortlisted_candidates', []) or state.get('shortlisted_candidates', [])
        job_req = context.get("job_requirements", {})
        
        return f"""
**TASK: INTERVIEW QUESTION GENERATION**

Create tailored interview questions for {len(shortlisted)} shortlisted candidates for the {job_req.get('title', 'Unknown Position')} role.

**QUESTION CATEGORIES:**
• Technical Questions: Assess technical competency and problem-solving
• Behavioral Questions: Evaluate soft skills and cultural fit
• Situational Questions: Test decision-making and crisis management
• Role-specific Questions: Target specific job requirements and scenarios

**CUSTOMIZATION REQUIREMENTS:**
• Tailor questions to each candidate's background and strengths
• Include questions that explore identified skill gaps
• Vary difficulty levels based on candidate experience
• Provide expected answer points and evaluation criteria

**CULTURAL CONSIDERATIONS:**
• Respect Mongolian cultural context and communication styles
• Include questions that assess cross-cultural collaboration skills
• Consider local market and business environment knowledge

**OUTPUT FORMAT:**
• Structured question sets for each candidate
• Difficulty ratings and time estimates
• Evaluation rubrics and scoring guidelines
• Alternative questions for follow-up
**IMPORTANT: All interview questions and content must be in English.**
"""
    
    def _generate_email_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate email agent specific prompt"""
        shortlisted = getattr(state, 'shortlisted_candidates', []) or state.get('shortlisted_candidates', [])
        all_candidates = getattr(state, 'candidate_scores', []) or state.get('candidate_scores', [])
        job_req = context.get("job_requirements", {})
        
        rejected_count = len(all_candidates) - len(shortlisted)
        
        return f"""
**TASK: EMAIL COMMUNICATION DRAFTING**

Create personalized email communications for {len(shortlisted)} shortlisted candidates (interviews) and {rejected_count} rejected candidates.

**EMAIL CATEGORIES:**
• Interview Invitations: For shortlisted candidates
• Polite Rejections: For non-shortlisted candidates
• Acknowledgment Emails: Confirming receipt of applications

**PERSONALIZATION REQUIREMENTS:**
• Include candidate's name and specific qualifications mentioned
• Reference specific strengths identified in evaluation
• Maintain professional yet warm and encouraging tone
• Include relevant next steps and timeline information

**CULTURAL AND LINGUISTIC CONSIDERATIONS:**
• Professional communication standards for Mongolian business context
• Respectful and encouraging language for rejected candidates
• Clear and actionable information for interview candidates
• Bilingual capability (prepare versions in both English and Mongolian)

**COMPANY CONTEXT:**
• Position: {job_req.get('title', 'Unknown Position')}
• Company: {job_req.get('industry', 'Company')}
• Professional branding and tone consistency

**OUTPUT FORMAT:**
• Complete email drafts with subject lines
• Personalized content for each recipient
• Alternative versions for different scenarios
• Scheduling and logistics information where applicable
**IMPORTANT: All email content must be generated in English.**
"""
    
    def _generate_generic_prompt(self, state: AgentState, context: Dict[str, Any]) -> str:
        """Generate generic prompt for unknown agent types"""
        return f"""
**TASK: {self.agent_type.upper()} PROCESSING**

Process the current workflow state according to your specialized function.

**CONTEXT:** {context.get("summary", "No context available")}

**REQUIREMENTS:**
• Follow established data schemas and formats
• Maintain consistency with previous agent outputs
• Provide detailed reasoning for decisions
• Include confidence scores and metadata
• Generate all output in English only

**IMPORTANT: All output must be in English regardless of input language.**

Please proceed with your specialized processing task.
""" 