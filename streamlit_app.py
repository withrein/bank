#!/usr/bin/env python3
"""
HR Multi-Agent System - Streamlit Web Interface
Comprehensive web interface with bilingual support and real-time monitoring
"""

import os
import time
import logging
import json
import tempfile
import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Khan Bank - AI HR System",
    page_icon="🏦",
    layout="wide"
)

from models import JobDescription, AgentState, ParsedCV, CandidateScore, InterviewQuestion
from workflow import HRWorkflow
from config import Config
from utils import save_json_output, create_output_directory, extract_text_from_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitHRApp:
    """Comprehensive Streamlit interface for HR Multi-Agent System"""
    
    def __init__(self):
        """Initialize the Streamlit HR app"""
        # Setup output directory
        self.setup_output_directory()
        
        # Initialize session state
        self.initialize_session_state()
        
        # Initialize config in session state if not already there
        if "config" not in st.session_state:
            st.session_state.config = {
                "MODEL_PROVIDER": Config.MODEL_PROVIDER,
                "TEMPERATURE": Config.TEMPERATURE,
                "MAX_TOKENS": Config.MAX_TOKENS,
                "MAX_CANDIDATES_TO_SHORTLIST": Config.MAX_CANDIDATES_TO_SHORTLIST,
                "MINIMUM_SCORE_THRESHOLD": Config.MINIMUM_SCORE_THRESHOLD,
                "DEFAULT_LANGUAGE": Config.DEFAULT_LANGUAGE
            }
        
    def setup_output_directory(self):
        """Setup output directory for results"""
        self.output_dir = create_output_directory()
        logger.info(f"Output directory: {self.output_dir}")
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'cv_files' not in st.session_state:
            st.session_state.cv_files = []
        if 'job_description' not in st.session_state:
            st.session_state.job_description = None
        if 'current_state' not in st.session_state:
            st.session_state.current_state = None
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "idle"
        if 'workflow_results' not in st.session_state:
            st.session_state.workflow_results = None
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = "mn"  # Default to Mongolian
        if 'output_language' not in st.session_state:
            st.session_state.output_language = "mn"  # For outputs selection
    
    def process_uploaded_files(self, uploaded_files) -> Tuple[List[str], str]:
        """Process uploaded CV files and return file paths"""
        if not uploaded_files:
            return [], "❌ No files uploaded"
        
        cv_files = []
        status_messages = []
        
        try:
            # Create temporary directory for uploaded files
            temp_dir = tempfile.mkdtemp()
            
            for uploaded_file in uploaded_files:
                # Check file extension
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext not in Config.ALLOWED_CV_FORMATS:
                    status_messages.append(f"⚠️ Skipped {uploaded_file.name} (unsupported format)")
                    continue
                
                # Save uploaded file to temp directory
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                cv_files.append(temp_path)
                status_messages.append(f"✅ {uploaded_file.name}")
            
            if cv_files:
                st.session_state.cv_files = cv_files
                status_msg = f"**Uploaded {len(cv_files)} CV files:**\n" + "\n".join(status_messages)
            else:
                status_msg = "❌ No valid CV files found"
                
        except Exception as e:
            status_msg = f"❌ Error processing files: {str(e)}"
            logger.error(f"File processing error: {e}")
        
        return cv_files, status_msg
    
    def create_job_description(self, form_data: Dict[str, Any]) -> Tuple[JobDescription, str]:
        """Create job description object from form inputs"""
        try:
            # Parse skills (comma-separated)
            req_skills = [skill.strip() for skill in form_data['required_skills'].split(',') if skill.strip()]
            pref_skills = [skill.strip() for skill in form_data['preferred_skills'].split(',') if skill.strip()]
            edu_req = [req.strip() for req in form_data['education_req'].split(',') if req.strip()]
            
            job_desc = JobDescription(
                title=form_data['title'] or "Untitled Position",
                company=form_data['company'] or "Company",
                location=form_data['location'],
                required_skills=req_skills,
                preferred_skills=pref_skills,
                min_experience=form_data['min_experience'] if form_data['min_experience'] > 0 else None,
                education_requirements=edu_req,
                job_type=form_data['job_type'],
                salary_range=form_data['salary_range'],
                description=form_data['description'] or "No description provided"
            )
            
            st.session_state.job_description = job_desc
            return job_desc, "✅ Job description created successfully"
            
        except Exception as e:
            error_msg = f"❌ Error creating job description: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def run_workflow_process(self) -> str:
        """Run the HR workflow process"""
        try:
            if not st.session_state.cv_files:
                return "❌ Error: No CV files provided"
            
            if not st.session_state.job_description:
                return "❌ Error: No job description provided"
            
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get current settings from session state if available
            config_updates = {}
            if "config" in st.session_state:
                config_updates = st.session_state.config
            
            # Create workflow
            workflow = HRWorkflow()
            
            # Update workflow configuration with latest settings
            if config_updates:
                workflow.update_config_values(config_updates)
                status_text.text("⚙️ Applied custom settings...")
                time.sleep(0.5)
            
            # Log configuration being used
            logging.info(f"Running workflow with: Max Candidates={Config.MAX_CANDIDATES_TO_SHORTLIST}, Min Score={Config.MINIMUM_SCORE_THRESHOLD}")
            
            # Set processing status
            st.session_state.processing_status = "processing"
            progress_bar.progress(10)
            status_text.text("🔍 Parsing CVs...")
            
            # Run workflow
            result_state = workflow.run_workflow(
                st.session_state.job_description,
                st.session_state.cv_files
            )
            
            progress_bar.progress(80)
            status_text.text("📊 Processing results...")
            
            # Ensure we have a proper result state
            if not result_state:
                raise Exception("Workflow returned empty result state")
            
            # Store results in session state with better error handling
            workflow_results = {}
            
            # Convert result objects to dictionaries for JSON serialization
            if hasattr(result_state, "parsed_cvs") and result_state.parsed_cvs:
                workflow_results['parsed_cvs'] = [cv.model_dump() if hasattr(cv, 'model_dump') else cv for cv in result_state.parsed_cvs]
            else:
                workflow_results['parsed_cvs'] = []
            
            if hasattr(result_state, "candidate_scores") and result_state.candidate_scores:
                workflow_results['candidate_scores'] = [score.model_dump() if hasattr(score, 'model_dump') else score for score in result_state.candidate_scores]
            else:
                workflow_results['candidate_scores'] = []
            
            if hasattr(result_state, "shortlisted_candidates") and result_state.shortlisted_candidates:
                workflow_results['shortlisted_candidates'] = [candidate.model_dump() if hasattr(candidate, 'model_dump') else candidate for candidate in result_state.shortlisted_candidates]
            else:
                workflow_results['shortlisted_candidates'] = []
            
            if hasattr(result_state, "interview_questions") and result_state.interview_questions:
                workflow_results['interview_questions'] = {}
                for name, questions in result_state.interview_questions.items():
                    workflow_results['interview_questions'][name] = questions.model_dump() if hasattr(questions, 'model_dump') else questions
            else:
                workflow_results['interview_questions'] = {}
            
            if hasattr(result_state, "email_drafts") and result_state.email_drafts:
                workflow_results['email_drafts'] = [email.model_dump() if hasattr(email, 'model_dump') else email for email in result_state.email_drafts]
            else:
                workflow_results['email_drafts'] = []
            
            if hasattr(result_state, "errors") and result_state.errors:
                workflow_results['errors'] = result_state.errors
            else:
                workflow_results['errors'] = []
            
            # Add metadata
            workflow_results['processing_status'] = 'completed'
            workflow_results['current_step'] = 'finalized'
            workflow_results['timestamp'] = datetime.now().isoformat()
            workflow_results['config_used'] = {
                'max_candidates': Config.MAX_CANDIDATES_TO_SHORTLIST,
                'min_score_threshold': Config.MINIMUM_SCORE_THRESHOLD,
                'model_provider': Config.MODEL_PROVIDER
            }
            
            # Store in session state
            st.session_state.workflow_results = workflow_results
            
            # Update processing status
            st.session_state.processing_status = "completed"
            progress_bar.progress(100)
            status_text.text("✅ Workflow completed!")
            
            # Log successful completion
            logging.info(f"Workflow completed successfully. Results stored in session state with {len(workflow_results['candidate_scores'])} candidates scored")
            
            return "✅ Workflow completed successfully"
            
        except Exception as e:
            logging.error(f"Error running workflow: {e}")
            st.session_state.processing_status = "error"
            return f"❌ Error: {str(e)}"
    
    def display_status_report(self, results):
        """Display workflow status report"""
        st.subheader("📊 Workflow статусын тайлан")
        
        # Helper function to get values from either dict or object
        def get_value(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        parsed_cvs = get_value(results, 'parsed_cvs', [])
        candidate_scores = get_value(results, 'candidate_scores', [])
        shortlisted_candidates = get_value(results, 'shortlisted_candidates', [])
        email_drafts = get_value(results, 'email_drafts', [])
        
        with col1:
            st.metric("CV боловсруулсан", len(parsed_cvs) if parsed_cvs else 0)
        with col2:
            st.metric("Нэр дэвшигч оноолсон", len(candidate_scores) if candidate_scores else 0)
        with col3:
            st.metric("Сонгогдсон", len(shortlisted_candidates) if shortlisted_candidates else 0)
        with col4:
            st.metric("Email ноорог", len(email_drafts) if email_drafts else 0)
        
        # Processing details
        st.write("**Process хийсэн дэлгэрэнгүй:**")
        st.write(f"- **Status:** {get_value(results, 'processing_status', 'Тодорхойгүй')}")
        st.write(f"- **Одоогийн алхам:** {get_value(results, 'current_step', 'Тодорхойгүй')}")
        st.write(f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Errors (if any)
        errors = get_value(results, 'errors', [])
        if errors:
            st.error("⚠️ **Алдаа гарсан:**")
            for error in errors:
                st.write(f"- {error}")
        
        # Job details
        job_description = get_value(results, 'job_description')
        if job_description:
            st.write("**Ажлын байрны дэлгэрэнгүй:**")
            st.write(f"- **Албан тушаал:** {get_value(job_description, 'title', 'Тодорхойгүй')}")
            st.write(f"- **Компани:** {get_value(job_description, 'company', 'Тодорхойгүй')}")
            location = get_value(job_description, 'location')
            if location:
                st.write(f"- **Байршил:** {location}")
            required_skills = get_value(job_description, 'required_skills', [])
            if required_skills:
                st.write(f"- **Шаардлагатай ур чадвар:** {', '.join(required_skills)}")
    
    def display_candidates_table(self, results):
        """Display candidates scoring table"""
        st.subheader("👥 Нэр дэвшигчдийн оноо")
        
        # Helper function to get values from either dict or object
        def get_value(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        candidate_scores = get_value(results, 'candidate_scores', [])
        shortlisted_candidates = get_value(results, 'shortlisted_candidates', [])
        
        if not candidate_scores:
            st.warning("Нэр дэвшигчдийн оноо байхгүй байна")
            return
        
        # Create DataFrame
        data = []
        shortlisted_names = [get_value(sc, 'candidate_name', '') for sc in shortlisted_candidates]
        
        for score in candidate_scores:
            candidate_name = get_value(score, 'candidate_name', 'Unknown')
            matched_skills = get_value(score, 'matched_skills', [])
            
            data.append({
                "Candidate": candidate_name,
                "Overall Score": round(get_value(score, 'overall_score', 0), 1),
                "Skills Match": round(get_value(score, 'skills_match_score', 0), 1),
                "Experience": round(get_value(score, 'experience_score', 0), 1),
                "Education": round(get_value(score, 'education_score', 0), 1),
                "Recommendation": get_value(score, 'recommendation', 'Unknown'),
                "Matched Skills": ", ".join(matched_skills[:3]) + ("..." if len(matched_skills) > 3 else ""),
                "Status": "✅ Shortlisted" if candidate_name in shortlisted_names else "❌ Not Selected"
            })
        
        df = pd.DataFrame(data)
        
        # Display table with styling
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Overall Score": st.column_config.ProgressColumn(
                    "Overall Score",
                    help="Overall matching score",
                    min_value=0,
                    max_value=100,
                ),
                "Skills Match": st.column_config.ProgressColumn(
                    "Skills Match",
                    help="Skills matching percentage",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
        
        # Score distribution chart
        st.subheader("📈 Score Distribution")
        
        scores = [get_value(score, 'overall_score', 0) for score in candidate_scores]
        fig = px.histogram(
            x=scores,
            nbins=10,
            title="Distribution of Candidate Scores",
            labels={'x': 'Overall Score', 'y': 'Number of Candidates'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_interview_questions(self, results):
        """Display interview questions"""
        st.subheader("🎤 Interview асуулгууд")
        
        # Helper function to get values from either dict or object
        def get_value(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        interview_questions = get_value(results, 'interview_questions', {})
        
        if not interview_questions:
            st.warning("Interview асуулгууд байхгүй байна")
            return
        
        # Select candidate
        candidate_names = list(interview_questions.keys())
        selected_candidate = st.selectbox("Select Candidate:", candidate_names)
        
        if selected_candidate:
            questions = interview_questions[selected_candidate]
            
            # Question summary
            col1, col2, col3, col4 = st.columns(4)
            
            technical_questions = get_value(questions, 'technical_questions', [])
            behavioral_questions = get_value(questions, 'behavioral_questions', [])
            role_specific_questions = get_value(questions, 'role_specific_questions', [])
            total_questions = get_value(questions, 'total_questions', len(technical_questions) + len(behavioral_questions) + len(role_specific_questions))
            
            with col1:
                st.metric("Total Questions", total_questions)
            with col2:
                st.metric("Technical", len(technical_questions))
            with col3:
                st.metric("Behavioral", len(behavioral_questions))
            with col4:
                st.metric("Role-specific", len(role_specific_questions))
            
            # Display questions by category
            tab1, tab2, tab3 = st.tabs(["Technical", "Behavioral", "Role-specific"])
            
            with tab1:
                if technical_questions:
                    for i, q in enumerate(technical_questions, 1):
                        difficulty = get_value(q, 'difficulty', 'Unknown')
                        question_text = get_value(q, 'question', 'No question text')
                        expected_answer_points = get_value(q, 'expected_answer_points', [])
                        
                        with st.expander(f"Technical Question {i} - {difficulty.title()}"):
                            st.write(f"**Question:** {question_text}")
                            if expected_answer_points:
                                st.write("**Expected Answer Points:**")
                                for point in expected_answer_points:
                                    st.write(f"- {point}")
                else:
                    st.info("No technical questions generated")
            
            with tab2:
                if behavioral_questions:
                    for i, q in enumerate(behavioral_questions, 1):
                        difficulty = get_value(q, 'difficulty', 'Unknown')
                        question_text = get_value(q, 'question', 'No question text')
                        expected_answer_points = get_value(q, 'expected_answer_points', [])
                        
                        with st.expander(f"Behavioral Question {i} - {difficulty.title()}"):
                            st.write(f"**Question:** {question_text}")
                            if expected_answer_points:
                                st.write("**Expected Answer Points:**")
                                for point in expected_answer_points:
                                    st.write(f"- {point}")
                else:
                    st.info("No behavioral questions generated")
            
            with tab3:
                if role_specific_questions:
                    for i, q in enumerate(role_specific_questions, 1):
                        difficulty = get_value(q, 'difficulty', 'Unknown')
                        question_text = get_value(q, 'question', 'No question text')
                        expected_answer_points = get_value(q, 'expected_answer_points', [])
                        
                        with st.expander(f"Role-specific Question {i} - {difficulty.title()}"):
                            st.write(f"**Question:** {question_text}")
                            if expected_answer_points:
                                st.write("**Expected Answer Points:**")
                                for point in expected_answer_points:
                                    st.write(f"- {point}")
                else:
                    st.info("No role-specific questions generated")
    
    def display_email_drafts(self, results):
        """Display email drafts"""
        st.subheader("📧 Email ноорог")
        
        # Helper function to get values from either dict or object
        def get_value(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        email_drafts = get_value(results, 'email_drafts', [])
        
        if not email_drafts:
            st.warning("Email ноорог байхгүй байна")
            return
        
        # Group emails by type
        email_types = {}
        for email in email_drafts:
            email_type = get_value(email, 'email_type', 'unknown')
            if email_type not in email_types:
                email_types[email_type] = []
            email_types[email_type].append(email)
        
        # Email type summary
        if email_types:
            cols = st.columns(len(email_types))
            for i, (email_type, emails) in enumerate(email_types.items()):
                with cols[i]:
                    st.metric(email_type.replace('_', ' ').title(), len(emails))
        
        # Display emails by type
        if email_types:
            tabs = st.tabs(list(email_types.keys()))
            
            for tab, (email_type, emails) in zip(tabs, email_types.items()):
                with tab:
                    for email in emails:
                        recipient_name = get_value(email, 'recipient_name', 'Unknown')
                        recipient_email = get_value(email, 'recipient_email', 'Unknown')
                        subject = get_value(email, 'subject', 'No subject')
                        body = get_value(email, 'body', 'No content')
                        
                        with st.expander(f"Email to {recipient_name}"):
                            st.write(f"**To:** {recipient_email}")
                            st.write(f"**Subject:** {subject}")
                            st.write("**Body:**")
                            st.write(body)
    
    def display_analytics(self, results):
        """Display analytics and insights"""
        st.subheader("📊 Analytics & Insights")
        
        # Helper function to get values from either dict or object
        def get_value(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        candidate_scores = get_value(results, 'candidate_scores', [])
        
        if not candidate_scores:
            st.warning("No analytics data available")
            return
        
        # Score statistics
        scores = [get_value(score, 'overall_score', 0) for score in candidate_scores]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{avg_score:.1f}")
            with col2:
                st.metric("Highest Score", f"{max_score:.1f}")
            with col3:
                st.metric("Lowest Score", f"{min_score:.1f}")
            
            # Score ranges
            high_performers = len([s for s in scores if s >= 80])
            good_performers = len([s for s in scores if 70 <= s < 80])
            average_performers = len([s for s in scores if 60 <= s < 70])
            below_average = len([s for s in scores if s < 60])
            
            # Performance distribution chart
            performance_data = {
                'Performance Level': ['High (80+)', 'Good (70-79)', 'Average (60-69)', 'Below Average (<60)'],
                'Count': [high_performers, good_performers, average_performers, below_average]
            }
            
            fig = px.pie(
                values=performance_data['Count'],
                names=performance_data['Performance Level'],
                title="Performance Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills analysis
        all_skills = set()
        missing_skills = set()
        for score in candidate_scores:
            matched_skills = get_value(score, 'matched_skills', [])
            missing_skills_candidate = get_value(score, 'missing_skills', [])
            all_skills.update(matched_skills)
            missing_skills.update(missing_skills_candidate)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Common Skills Found:**")
            if all_skills:
                for skill in list(all_skills)[:10]:
                    st.write(f"- {skill}")
        
        with col2:
            st.write("**Most Common Missing Skills:**")
            if missing_skills:
                for skill in list(missing_skills)[:10]:
                    st.write(f"- {skill}")
        
        # Top candidates summary
        top_candidates = sorted(candidate_scores, key=lambda x: get_value(x, 'overall_score', 0), reverse=True)[:5]
        st.write("**Top 5 Candidates:**")
        for i, candidate in enumerate(top_candidates, 1):
            candidate_name = get_value(candidate, 'candidate_name', 'Unknown')
            overall_score = get_value(candidate, 'overall_score', 0)
            recommendation = get_value(candidate, 'recommendation', 'No recommendation')
            st.write(f"{i}. **{candidate_name}** - {overall_score:.1f} points")
            st.write(f"   {recommendation}")
    
    def export_results(self, format_type: str) -> Optional[bytes]:
        """Export results in specified format"""
        try:
            results = st.session_state.workflow_results
            
            if not results:
                st.error("Export хийх өгөгдөл байхгүй байна")
                return None
            
            # Helper function to get values from either dict or object
            def get_value(obj, key, default=None):
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                return default
            
            # Helper function to serialize objects
            def serialize_object(obj):
                if hasattr(obj, 'dict'):
                    return obj.dict()
                elif hasattr(obj, 'model_dump'):
                    return obj.model_dump()
                elif isinstance(obj, dict):
                    return obj
                else:
                    return str(obj)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "JSON":
                # Export as JSON
                job_description = get_value(results, 'job_description')
                candidate_scores = get_value(results, 'candidate_scores', [])
                shortlisted_candidates = get_value(results, 'shortlisted_candidates', [])
                interview_questions = get_value(results, 'interview_questions', {})
                email_drafts = get_value(results, 'email_drafts', [])
                
                export_data = {
                    "job_description": serialize_object(job_description) if job_description else None,
                    "candidate_scores": [serialize_object(score) for score in candidate_scores],
                    "shortlisted_candidates": [serialize_object(score) for score in shortlisted_candidates],
                    "interview_questions": {name: serialize_object(questions) for name, questions in interview_questions.items()},
                    "email_drafts": [serialize_object(email) for email in email_drafts],
                    "export_timestamp": timestamp
                }
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                return json_str.encode('utf-8')
            
            elif format_type == "Excel":
                # Export as Excel
                output = io.BytesIO()
                
                candidate_scores = get_value(results, 'candidate_scores', [])
                shortlisted_candidates = get_value(results, 'shortlisted_candidates', [])
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Candidates sheet
                    if candidate_scores:
                        candidates_data = []
                        for score in candidate_scores:
                            matched_skills = get_value(score, 'matched_skills', [])
                            missing_skills = get_value(score, 'missing_skills', [])
                            
                            candidates_data.append({
                                "Candidate Name": get_value(score, 'candidate_name', 'Unknown'),
                                "File Name": get_value(score, 'file_name', 'Unknown'),
                                "Overall Score": get_value(score, 'overall_score', 0),
                                "Skills Match": get_value(score, 'skills_match_score', 0),
                                "Experience Score": get_value(score, 'experience_score', 0),
                                "Education Score": get_value(score, 'education_score', 0),
                                "Matched Skills": ", ".join(matched_skills) if matched_skills else "",
                                "Missing Skills": ", ".join(missing_skills) if missing_skills else "",
                                "Recommendation": get_value(score, 'recommendation', 'Unknown'),
                                "Reasoning": get_value(score, 'reasoning', 'No reasoning')
                            })
                        
                        df_candidates = pd.DataFrame(candidates_data)
                        df_candidates.to_excel(writer, sheet_name='Candidates', index=False)
                    
                    # Shortlisted sheet
                    if shortlisted_candidates:
                        shortlisted_data = []
                        for score in shortlisted_candidates:
                            strengths = get_value(score, 'strengths', [])
                            
                            shortlisted_data.append({
                                "Candidate Name": get_value(score, 'candidate_name', 'Unknown'),
                                "Overall Score": get_value(score, 'overall_score', 0),
                                "Strengths": ", ".join(strengths) if strengths else "",
                                "Recommendation": get_value(score, 'recommendation', 'Unknown')
                            })
                        
                        df_shortlisted = pd.DataFrame(shortlisted_data)
                        df_shortlisted.to_excel(writer, sheet_name='Shortlisted', index=False)
                
                return output.getvalue()
            
            return None
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            logger.error(f"Export error: {e}")
            return None
    
    def render_sidebar(self):
        """Render sidebar with navigation and settings"""
        st.sidebar.title("🤖 HR Multi-Agent System")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Шилжих хэсэг",
            ["🏠 Нүүр хуудас", "📁 CV файл оруулах", "💼 Ажлын байрны тайлбар", "⚙️ Process хийх", "📊 Үр дүн", "🔧 Тохиргоо"]
        )
        
        st.sidebar.markdown("---")
        
        # Quick status
        st.sidebar.subheader("📈 Хурдан харах")
        st.sidebar.write(f"**CV-ууд:** {len(st.session_state.cv_files)}")
        st.sidebar.write(f"**Ажлын байрны тайлбар:** {'✅' if st.session_state.job_description else '❌'}")
        st.sidebar.write(f"**Status:** {st.session_state.processing_status}")
        
        if st.session_state.workflow_results:
            results = st.session_state.workflow_results
            
            # Helper function to get values from either dict or object
            def get_value(obj, key, default=None):
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                return default
            
            candidate_scores = get_value(results, 'candidate_scores', [])
            shortlisted_candidates = get_value(results, 'shortlisted_candidates', [])
            
            st.sidebar.write(f"**Нэр дэвшигчид:** {len(candidate_scores) if candidate_scores else 0}")
            st.sidebar.write(f"**Сонгогдсон:** {len(shortlisted_candidates) if shortlisted_candidates else 0}")
        
        return page
    
    def render_home_page(self):
        """Render home page"""
        st.title("🤖 HR Multi-Agent System")
        st.markdown("### Автомат HR ажилтан авах систем / Automated HR Recruitment System")
        
        st.markdown("""
        HR Multi-Agent System-д тавтай морил! Энэ ухаалаг platform нь AI-powered agent-уудыг ашиглан 
        ажил олгогчийн бүрэн workflow-г автоматжуулдаг.
        
        **Онцлог хэрэгслүүд:**
        - 🔍 **CV Analysis:** Нэр дэвшигчийн мэдээллийг автоматаар task хийх ба гаргаж авах
        - 📊 **Intelligent Scoring:** Ажлын шаардлагын эсрэг AI-powered нэр дэвшигчийн үнэлгээ  
        - 🎯 **Smart Shortlisting:** Шилдэг нэр дэвшигчдийг автоматаар сонгох
        - 🎤 **Interview Асуулга:** Нэр дэвшигч бүрт зориулсан тусгай асуулгууд
        - 📧 **Email Automation:** Хувийн харилцааны template-ууд
        - 🌍 **Bilingual Support:** Монгол болон англи хэлний processing
        """)
        
        # Quick start guide
        st.subheader("🚀 Хурдан эхлэх заавар")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Алхам 1: CV файлууд оруулах**
            - "📁 CV файл оруулах" хэсэгт орно уу
            - PDF, DOCX, эсвэл TXT файлууд upload хийнэ үү
            - System нь хоёр хэлийн баримт бичгийг дэмждэг
            """)
        
        with col2:
            st.markdown("""
            **Алхам 2: Ажлын байрны тайлбар**
            - "💼 Ажлын байрны тайлбар" хэсэгт орно уу
            - Албан тушаалын шаардлагыг тодорхойлно уу
            - Ур чадвар болон туршлагын шалгуур тавина уу
            """)
        
        with col3:
            st.markdown("""
            **Алхам 3: Process ба үр дүн**
            - "⚙️ Process хийх" хэсэгт орж workflow ажиллуулна уу
            - "📊 Үр дүн" хэсэгт үр дүнг харна уу
            - Өгөгдлийг олон төрлийн format-аар export хийнэ үү
            """)
        
        # System status
        if st.session_state.workflow_results:
            st.success("✅ System бэлэн байна! Танд боловсруулсан үр дүн байна.")
        elif st.session_state.cv_files and st.session_state.job_description:
            st.info("🔄 Process хийхэд бэлэн! 'Process хийх' хэсэгт орж workflow эхлүүлнэ үү.")
        else:
            st.warning("⚠️ Тохиргоо шаардлагатай: CV файлууд upload хийж, ажлын байрны тайлбар үүсгэнэ үү.")
    
    def render_upload_page(self):
        """Render file upload page"""
        st.title("📁 CV файлууд оруулах")
        st.markdown("### Нэр дэвшигчдийн CV файлуудыг боловсруулахаар upload хийнэ үү")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "CV файлууд сонгох",
            type=Config.ALLOWED_CV_FORMATS,
            accept_multiple_files=True,
            help="Дэмжигдэх format-ууд: PDF, DOCX, DOC, TXT"
        )
        
        if uploaded_files:
            cv_files, status_msg = self.process_uploaded_files(uploaded_files)
            st.markdown(status_msg)
            
            if cv_files:
                st.success(f"✅ {len(cv_files)} файл амжилттай upload хийлээ")
        
        # Display current files
        if st.session_state.cv_files:
            st.subheader("📄 Одоогийн файлууд")
            for i, file_path in enumerate(st.session_state.cv_files, 1):
                st.write(f"{i}. {os.path.basename(file_path)}")
            
            if st.button("🗑️ Бүх файлуудыг арилгах"):
                st.session_state.cv_files = []
                st.rerun()
    
    def render_job_description_page(self):
        """Render job description page"""
        st.title("💼 Ажлын байрны тайлбар")
        st.markdown("### Албан тушаалын шаардлага болон шалгуурыг тодорхойлно уу")
        
        with st.form("job_description_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Ажлын байрны нэр *", placeholder="жнь: Senior Software Engineer")
                company = st.text_input("Компани *", placeholder="жнь: Хаан банк")
                location = st.text_input("Байршил", placeholder="жнь: Улаанбаатар, Монгол")
                job_type = st.selectbox("Ажлын төрөл", ["Бүтэн цагийн", "Хагас цагийн", "Гэрээт", "Түр зуурын"])
            
            with col2:
                min_experience = st.number_input("Хамгийн бага туршлага (жил)", min_value=0, value=0)
                salary_range = st.text_input("Цалингийн хүрээ", placeholder="жнь: $80,000 - $120,000")
                
            required_skills = st.text_area(
                "Шаардлагатай ур чадвар (таслалаар тусгаарлана) *",
                placeholder="жнь: Python, JavaScript, React, SQL",
                height=100
            )
            
            preferred_skills = st.text_area(
                "Давуу талтай ур чадвар (таслалаар тусгаарлана)",
                placeholder="жнь: AWS, Docker, Kubernetes",
                height=100
            )
            
            education_req = st.text_area(
                "Боловсролын шаардлага",
                placeholder="жнь: Computer Science-ийн бакалаврын зэрэг",
                height=80
            )
            
            description = st.text_area(
                "Ажлын байрны дэлгэрэнгүй тайлбар *",
                placeholder="Ажлын байрны дэлгэрэнгүй тайлбарыг оруулна уу...",
                height=200
            )
            
            submitted = st.form_submit_button("💼 Ажлын байрны тайлбар үүсгэх")
            
            if submitted:
                form_data = {
                    'title': title,
                    'company': company,
                    'location': location,
                    'required_skills': required_skills,
                    'preferred_skills': preferred_skills,
                    'min_experience': min_experience,
                    'education_req': education_req,
                    'job_type': job_type,
                    'salary_range': salary_range,
                    'description': description
                }
                
                job_desc, status_msg = self.create_job_description(form_data)
                
                if job_desc:
                    st.success(status_msg)
                else:
                    st.error(status_msg)
        
        # Display current job description
        if st.session_state.job_description:
            st.subheader("📋 Current Job Description")
            job = st.session_state.job_description
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Title:** {job.title}")
                st.write(f"**Company:** {job.company}")
                if job.location:
                    st.write(f"**Location:** {job.location}")
                st.write(f"**Type:** {job.job_type}")
            
            with col2:
                if job.min_experience:
                    st.write(f"**Min Experience:** {job.min_experience} years")
                if job.salary_range:
                    st.write(f"**Salary:** {job.salary_range}")
            
            if job.required_skills:
                st.write(f"**Required Skills:** {', '.join(job.required_skills)}")
            if job.preferred_skills:
                st.write(f"**Preferred Skills:** {', '.join(job.preferred_skills)}")
    
    def render_process_page(self):
        """Render processing page"""
        st.title("⚙️ Process Workflow")
        st.markdown("### HR Multi-Agent System ажиллуулах")
        
        # Prerequisites check
        ready_to_process = bool(st.session_state.cv_files and st.session_state.job_description)
        
        if not ready_to_process:
            st.warning("⚠️ **Урьдчилсан шаардлага хангагдаагүй:**")
            if not st.session_state.cv_files:
                st.write("- CV файлууд upload хийнэ үү")
            if not st.session_state.job_description:
                st.write("- Ажлын байрны тайлбар үүсгэнэ үү")
            return
        
        # Ready to process
        st.success("✅ **Process хийхэд бэлэн!**")
        st.write(f"- **CV файлууд:** {len(st.session_state.cv_files)}")
        st.write(f"- **Ажлын байр:** {st.session_state.job_description.title}")
        
        # Process button
        if st.button("🚀 HR Workflow эхлүүлэх", type="primary", use_container_width=True):
            with st.spinner("Боловсруулж байна..."):
                status_msg = self.run_workflow_process()
                
                if "successfully" in status_msg:
                    st.success(status_msg)
                    st.balloons()
                else:
                    st.error(status_msg)
        
        # Current status
        st.subheader("📊 Одоогийн төлөв")
        st.write(f"**Processing Status:** {st.session_state.processing_status}")
        
        if st.session_state.workflow_results:
            results = st.session_state.workflow_results
            self.display_status_report(results)
    
    def render_results_page(self):
        """Render results page"""
        st.title("📊 Үр дүнгийн Dashboard")
        st.markdown("### Process хийсэн үр дүнг харах ба шинжлэх")
        
        # Try to get results from session state first
        results = st.session_state.get('workflow_results', None)
        
        # If no results in session state, try to load from output files
        if not results:
            st.info("💾 Session state-ээс үр дүн олдсонгүй. Output файлуудаас уншиж байна...")
            results = self.load_results_from_files()
        
        if not results:
            st.warning("⚠️ Үр дүн байхгүй байна. Эхлээд workflow ажиллуулна уу.")
            
            # Show debug information
            with st.expander("🔍 Debug мэдээлэл"):
                st.write("**Session State Keys:**", list(st.session_state.keys()))
                st.write("**Output Directory:**", self.output_dir)
                
                # Check if output files exist
                output_files = [
                    "parsed_cvs.json",
                    "candidate_scores.json", 
                    "shortlisted_candidates.json",
                    "interview_questions.json",
                    "email_drafts.json"
                ]
                
                st.write("**Output файлууд:**")
                for file in output_files:
                    file_path = os.path.join(self.output_dir, file)
                    if os.path.exists(file_path):
                        st.write(f"✅ {file}")
                    else:
                        st.write(f"❌ {file}")
            return
        
        # Store results in session state for future use
        st.session_state.workflow_results = results
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Ерөнхий харагдац", "👥 Нэр дэвшигчид", "🎤 Interview асуулгууд", "📧 Email ноорог", "📈 Analytics"
        ])
        
        with tab1:
            self.display_status_report(results)
        
        with tab2:
            self.display_candidates_table(results)
        
        with tab3:
            self.display_interview_questions(results)
        
        with tab4:
            self.display_email_drafts(results)
        
        with tab5:
            self.display_analytics(results)
        
        # Export functionality
        st.markdown("---")
        st.subheader("📥 Үр дүн Export хийх")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 JSON болгон export хийх"):
                json_data = self.export_results("JSON")
                if json_data:
                    st.download_button(
                        label="JSON татаж авах",
                        data=json_data,
                        file_name=f"hr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            if st.button("📊 Excel болгон export хийх"):
                excel_data = self.export_results("Excel")
                if excel_data:
                    st.download_button(
                        label="Excel татаж авах",
                        data=excel_data,
                        file_name=f"hr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    def load_results_from_files(self):
        """Load results from output JSON files"""
        try:
            results = {}
            
            # Load parsed CVs
            parsed_cvs_path = os.path.join(self.output_dir, "parsed_cvs.json")
            if os.path.exists(parsed_cvs_path):
                with open(parsed_cvs_path, 'r', encoding='utf-8') as f:
                    results['parsed_cvs'] = json.load(f)
            
            # Load candidate scores
            scores_path = os.path.join(self.output_dir, "candidate_scores.json")
            if os.path.exists(scores_path):
                with open(scores_path, 'r', encoding='utf-8') as f:
                    results['candidate_scores'] = json.load(f)
            
            # Load shortlisted candidates
            shortlisted_path = os.path.join(self.output_dir, "shortlisted_candidates.json")
            if os.path.exists(shortlisted_path):
                with open(shortlisted_path, 'r', encoding='utf-8') as f:
                    results['shortlisted_candidates'] = json.load(f)
            
            # Load interview questions
            questions_path = os.path.join(self.output_dir, "interview_questions.json")
            if os.path.exists(questions_path):
                with open(questions_path, 'r', encoding='utf-8') as f:
                    results['interview_questions'] = json.load(f)
            
            # Load email drafts
            emails_path = os.path.join(self.output_dir, "email_drafts.json")
            if os.path.exists(emails_path):
                with open(emails_path, 'r', encoding='utf-8') as f:
                    results['email_drafts'] = json.load(f)
            
            # Load complete workflow state if available
            complete_state_path = os.path.join(self.output_dir, "complete_workflow_state.json")
            if os.path.exists(complete_state_path):
                with open(complete_state_path, 'r', encoding='utf-8') as f:
                    complete_state = json.load(f)
                    # Merge additional information from complete state
                    results.update(complete_state)
            
            results['errors'] = []
            results['processing_status'] = 'completed'
            results['current_step'] = 'finalized'
            
            if results:
                st.success(f"✅ Output файлуудаас үр дүн амжилттай уншив!")
                return results
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ Output файлуудаас үр дүн унших үед алдаа: {str(e)}")
            logger.error(f"Error loading results from files: {e}")
            return None
    
    def render_settings_page(self):
        """Render settings page"""
        st.title("🔧 Settings")
        st.markdown("### Configure system parameters")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_provider = st.selectbox(
                "Model Provider",
                ["openai", "gemini"],
                index=0 if Config.MODEL_PROVIDER == "openai" else 1,
                key="settings_model_provider"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=Config.TEMPERATURE,
                step=0.1,
                help="Controls randomness in AI responses",
                key="settings_temperature"
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=8000,
                value=Config.MAX_TOKENS,
                help="Maximum response length",
                key="settings_max_tokens"
            )
        
        # Processing settings
        st.subheader("⚙️ Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_candidates = st.number_input(
                "Max Candidates to Shortlist",
                min_value=1,
                max_value=20,
                value=Config.MAX_CANDIDATES_TO_SHORTLIST,
                key="settings_max_candidates"
            )
            
            min_score = st.number_input(
                "Minimum Score Threshold",
                min_value=0,
                max_value=100,
                value=Config.MINIMUM_SCORE_THRESHOLD,
                key="settings_min_score"
            )
        
        with col2:
            default_language = st.selectbox(
                "Default Language",
                ["mn", "en"],
                index=0 if Config.DEFAULT_LANGUAGE == "mn" else 1,
                key="settings_default_language"
            )
        
        # Save settings button
        if st.button("💾 Save Settings"):
            # Update Config values
            Config.MODEL_PROVIDER = model_provider
            Config.TEMPERATURE = temperature
            Config.MAX_TOKENS = max_tokens
            Config.MAX_CANDIDATES_TO_SHORTLIST = max_candidates
            Config.MINIMUM_SCORE_THRESHOLD = min_score
            Config.DEFAULT_LANGUAGE = default_language
            
            # Store in session state for persistence
            if "config" not in st.session_state:
                st.session_state.config = {}
            
            st.session_state.config["MODEL_PROVIDER"] = model_provider
            st.session_state.config["TEMPERATURE"] = temperature
            st.session_state.config["MAX_TOKENS"] = max_tokens
            st.session_state.config["MAX_CANDIDATES_TO_SHORTLIST"] = max_candidates
            st.session_state.config["MINIMUM_SCORE_THRESHOLD"] = min_score
            st.session_state.config["DEFAULT_LANGUAGE"] = default_language
            
            st.success("✅ Settings saved and applied successfully!")
        
        # System information
        st.subheader("ℹ️ System Information")
        st.write(f"**Output Directory:** {self.output_dir}")
        st.write(f"**Supported Formats:** {', '.join(Config.ALLOWED_CV_FORMATS)}")
        st.write(f"**Max File Size:** {Config.MAX_FILE_SIZE_MB} MB")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        page = self.render_sidebar()
        
        # Render selected page
        if page == "🏠 Нүүр хуудас":
            self.render_home_page()
        elif page == "📁 CV файл оруулах":
            self.render_upload_page()
        elif page == "💼 Ажлын байрны тайлбар":
            self.render_job_description_page()
        elif page == "⚙️ Process хийх":
            self.render_process_page()
        elif page == "📊 Үр дүн":
            self.render_results_page()
        elif page == "🔧 Тохиргоо":
            self.render_settings_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    app = StreamlitHRApp()
    app.run()

if __name__ == "__main__":
    main() 