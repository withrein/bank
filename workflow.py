from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models import AgentState, JobDescription
from agents.cv_parser_agent import CVParserAgent
from agents.scoring_agent import ScoringAgent
from agents.shortlisting_agent import ShortlistingAgent
from agents.interview_agent import InterviewAgent
from agents.email_agent import EmailAgent
from utils import create_output_directory, save_json_output

def get_state_value(state: Union[AgentState, Dict], key: str, default=None):
    """Get value from state, handling both AgentState objects and dicts"""
    if hasattr(state, key):
        return getattr(state, key)
    elif isinstance(state, dict):
        return state.get(key, default)
    else:
        return default

def set_state_value(state: Union[AgentState, Dict], key: str, value):
    """Set value in state, handling both AgentState objects and dicts"""
    if hasattr(state, key):
        setattr(state, key, value)
    elif isinstance(state, dict):
        state[key] = value

def append_state_error(state: Union[AgentState, Dict], error_msg: str):
    """Append error to state, handling both AgentState objects and dicts"""
    if hasattr(state, 'errors'):
        state.errors.append(error_msg)
    elif isinstance(state, dict):
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)

class HRWorkflow:
    """Main workflow orchestrator for the HR Multi-Agent System"""
    
    def __init__(self):
        self.cv_parser = CVParserAgent()
        self.scoring_agent = ScoringAgent()
        self.shortlisting_agent = ShortlistingAgent()
        self.interview_agent = InterviewAgent()
        self.email_agent = EmailAgent()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        # Setup memory for checkpointing
        self.memory = MemorySaver()
        
        # Compile the workflow
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("parse_cvs", self._parse_cvs_node)
        workflow.add_node("score_candidates", self._score_candidates_node)
        workflow.add_node("shortlist_candidates", self._shortlist_candidates_node)
        workflow.add_node("generate_questions", self._generate_questions_node)
        workflow.add_node("draft_emails", self._draft_emails_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define the workflow edges
        workflow.set_entry_point("parse_cvs")
        workflow.add_edge("parse_cvs", "score_candidates")
        workflow.add_edge("score_candidates", "shortlist_candidates")
        workflow.add_edge("shortlist_candidates", "generate_questions")
        workflow.add_edge("generate_questions", "draft_emails")
        workflow.add_edge("draft_emails", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        return workflow
    
    def _parse_cvs_node(self, state: AgentState) -> AgentState:
        """Node for CV parsing"""
        return self.cv_parser.process(state)
    
    def _score_candidates_node(self, state: AgentState) -> AgentState:
        """Node for candidate scoring"""
        return self.scoring_agent.process(state)
    
    def _shortlist_candidates_node(self, state: AgentState) -> AgentState:
        """Node for candidate shortlisting"""
        return self.shortlisting_agent.process(state)
    
    def _generate_questions_node(self, state: AgentState) -> AgentState:
        """Node for interview question generation"""
        return self.interview_agent.process(state)
    
    def _draft_emails_node(self, state: AgentState) -> AgentState:
        """Node for email drafting"""
        return self.email_agent.process(state)
    
    def _finalize_results_node(self, state: AgentState) -> AgentState:
        """Node for finalizing and saving results"""
        try:
            print("💾 Finalizing results and saving outputs...")
            
            # Create output directory
            output_dir = create_output_directory("outputs")
            
            # Save parsed CVs
            parsed_cvs = get_state_value(state, 'parsed_cvs')
            if parsed_cvs:
                save_json_output(
                    [cv.model_dump() for cv in parsed_cvs],
                    f"{output_dir}/parsed_cvs.json"
                )
            
            # Save candidate scores
            candidate_scores = get_state_value(state, 'candidate_scores')
            if candidate_scores:
                save_json_output(
                    [score.model_dump() for score in candidate_scores],
                    f"{output_dir}/candidate_scores.json"
                )
            
            # Save shortlisted candidates
            shortlisted_candidates = get_state_value(state, 'shortlisted_candidates')
            if shortlisted_candidates:
                save_json_output(
                    [candidate.model_dump() for candidate in shortlisted_candidates],
                    f"{output_dir}/shortlisted_candidates.json"
                )
            
            # Save interview questions
            interview_questions = get_state_value(state, 'interview_questions')
            if interview_questions:
                questions_dict = {}
                for name, questions in interview_questions.items():
                    questions_dict[name] = questions.model_dump()
                save_json_output(questions_dict, f"{output_dir}/interview_questions.json")
            
            # Save email drafts
            email_drafts = get_state_value(state, 'email_drafts')
            if email_drafts:
                save_json_output(
                    [email.model_dump() for email in email_drafts],
                    f"{output_dir}/email_drafts.json"
                )
            
            # Save complete workflow state
            try:
                if hasattr(state, 'model_dump'):
                    save_json_output(state.model_dump(), f"{output_dir}/complete_workflow_state.json")
                else:
                    # Convert AddableValuesDict to regular dict for JSON serialization
                    state_dict = dict(state)
                    save_json_output(state_dict, f"{output_dir}/complete_workflow_state.json")
            except Exception as state_save_error:
                print(f"⚠️ Could not save complete state: {state_save_error}")
                # Save a simplified version
                simple_state = {
                    "processing_status": state.get("processing_status", "unknown"),
                    "current_step": state.get("current_step", "unknown"),
                    "errors": state.get("errors", [])
                }
                save_json_output(simple_state, f"{output_dir}/workflow_status.json")
            
            set_state_value(state, "processing_status", "completed")
            set_state_value(state, "current_step", "completed")
            
            print(f"✅ Results saved to {output_dir}/")
            
        except Exception as e:
            error_msg = f"Error finalizing results: {str(e)}"
            print(f"❌ {error_msg}")
            append_state_error(state, error_msg)
            set_state_value(state, "processing_status", "failed")
        
        return state
    
    def run_workflow(self, job_description: JobDescription, cv_files: list) -> AgentState:
        """Run the complete HR workflow"""
        
        print("🚀 Starting HR Multi-Agent Workflow...")
        print(f"📋 Job: {job_description.title} at {job_description.company}")
        print(f"📄 Processing {len(cv_files)} CV files")
        print("=" * 60)
        
        # Initialize the state
        initial_state = AgentState(
            job_description=job_description,
            cv_files=cv_files,
            processing_status="running"
        )
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": "hr_workflow_001"}}
            final_state = self.app.invoke(initial_state, config=config)
            
            # Print final summary
            self._print_workflow_summary(final_state)
            
            return final_state
            
        except Exception as e:
            print(f"❌ Workflow execution failed: {str(e)}")
            append_state_error(initial_state, f"Workflow execution failed: {str(e)}")
            set_state_value(initial_state, "processing_status", "failed")
            return initial_state
    
    def _print_workflow_summary(self, state: AgentState):
        """Print a summary of the workflow results"""
        print("\n" + "=" * 60)
        print("🎉 HR WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        parsed_cvs = get_state_value(state, 'parsed_cvs')
        if parsed_cvs:
            print(f"📄 CVs Parsed: {len(parsed_cvs)}")
        
        candidate_scores = get_state_value(state, 'candidate_scores')
        if candidate_scores:
            print(f"📊 Candidates Scored: {len(candidate_scores)}")
            avg_score = sum(c.overall_score for c in candidate_scores) / len(candidate_scores)
            print(f"📈 Average Score: {avg_score:.1f}/100")
        
        shortlisted_candidates = get_state_value(state, 'shortlisted_candidates')
        if shortlisted_candidates:
            print(f"🎯 Candidates Shortlisted: {len(shortlisted_candidates)}")
            print("🏆 Top Candidates:")
            for i, candidate in enumerate(shortlisted_candidates[:3], 1):
                print(f"   {i}. {candidate.candidate_name} ({candidate.overall_score:.1f}/100)")
        
        interview_questions = get_state_value(state, 'interview_questions')
        if interview_questions:
            total_questions = sum(q.total_questions for q in interview_questions.values())
            print(f"❓ Interview Questions Generated: {total_questions}")
        
        email_drafts = get_state_value(state, 'email_drafts')
        if email_drafts:
            print(f"📧 Email Drafts Created: {len(email_drafts)}")
            email_types = {}
            for email in email_drafts:
                email_type = email.email_type
                email_types[email_type] = email_types.get(email_type, 0) + 1
            
            for email_type, count in email_types.items():
                print(f"   - {email_type.replace('_', ' ').title()}: {count}")
        
        errors = get_state_value(state, 'errors')
        if errors:
            print(f"⚠️  Errors Encountered: {len(errors)}")
            for error in errors:
                print(f"   - {error}")
        
        print(f"📁 Results saved to: outputs/")
        print("=" * 60)
    
    def get_workflow_status(self, thread_id: str = "hr_workflow_001") -> Dict[str, Any]:
        """Get the current status of a running workflow"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Get the latest state from memory
            state = self.app.get_state(config)
            
            if state and state.values:
                return {
                    "status": state.values.get("processing_status", "unknown"),
                    "current_step": state.values.get("current_step", "unknown"),
                    "errors": state.values.get("errors", []),
                    "progress": self._calculate_progress(state.values.get("current_step", "start"))
                }
            else:
                return {
                    "status": "not_started",
                    "current_step": "not_started",
                    "errors": [],
                    "progress": 0
                }
        except Exception as e:
            return {
                "status": "error",
                "current_step": "error",
                "errors": [str(e)],
                "progress": 0
            }
    
    def _calculate_progress(self, current_step: str) -> int:
        """Calculate workflow progress percentage"""
        steps = {
            "start": 0,
            "parsing_cvs": 10,
            "cvs_parsed": 20,
            "scoring_candidates": 30,
            "candidates_scored": 50,
            "shortlisting_candidates": 60,
            "candidates_shortlisted": 70,
            "generating_questions": 80,
            "questions_generated": 85,
            "drafting_emails": 90,
            "emails_drafted": 95,
            "completed": 100
        }
        
        return steps.get(current_step, 0) 