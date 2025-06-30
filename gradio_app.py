import gradio as gr
import os
import json
from typing import List, Tuple, Any
import pandas as pd

from models import JobDescription
from workflow import HRWorkflow, get_state_value
from config import Config
from utils import validate_file_format, get_file_size_mb

class GradioHRApp:
    """Gradio interface for the HR Multi-Agent System"""
    
    def __init__(self):
        self.workflow = HRWorkflow()
        self.current_results = None
    
    def run_hr_workflow(self, title: str, company: str, description: str, 
                       required_skills: str, cv_files: List[Any]) -> Tuple[str, str]:
        try:
            if not title or not company or not description:
                return "❌ Error: Job title, company, and description are required.", ""
            
            if not cv_files:
                return "❌ Error: Please upload at least one CV file.", ""
            
            # Process job description
            job_description = JobDescription(
                title=title.strip(),
                company=company.strip(),
                description=description.strip(),
                required_skills=[skill.strip() for skill in required_skills.split(',') if skill.strip()]
            )
            
            # Get file paths
            valid_files = []
            for file in cv_files:
                if file is not None:
                    valid_files.append(file.name)
            
            if not valid_files:
                return "❌ Error: No valid CV files found.", ""
            
            # Run workflow
            self.current_results = self.workflow.run_workflow(job_description, valid_files)
            
            # Generate summary
            summary = self._generate_summary()
            details = self._generate_details()
            
            return summary, details
            
        except Exception as e:
            return f"❌ Workflow failed: {str(e)}", ""
    
    def _generate_summary(self) -> str:
        if not self.current_results:
            return "No results available."
        
        state = self.current_results
        summary = "# 🎉 HR Workflow Results\n\n"
        
        processing_status = get_state_value(state, 'processing_status')
        if processing_status == "completed":
            summary += "✅ **Status**: Completed Successfully\n\n"
        
        parsed_cvs = get_state_value(state, 'parsed_cvs')
        if parsed_cvs:
            summary += f"📄 **CVs Processed**: {len(parsed_cvs)}\n"
        
        candidate_scores = get_state_value(state, 'candidate_scores')
        if candidate_scores:
            summary += f"📊 **Candidates Evaluated**: {len(candidate_scores)}\n"
            avg_score = sum(c.overall_score for c in candidate_scores) / len(candidate_scores)
            summary += f"📈 **Average Score**: {avg_score:.1f}/100\n"
        
        shortlisted_candidates = get_state_value(state, 'shortlisted_candidates')
        if shortlisted_candidates:
            summary += f"🎯 **Candidates Shortlisted**: {len(shortlisted_candidates)}\n"
            summary += "\n**Top Candidates:**\n"
            for i, candidate in enumerate(shortlisted_candidates[:3], 1):
                summary += f"{i}. {candidate.candidate_name} ({candidate.overall_score:.1f}/100)\n"
        
        return summary
    
    def _generate_details(self) -> str:
        if not self.current_results:
            return "No detailed results available."
        
        shortlisted_candidates = get_state_value(self.current_results, 'shortlisted_candidates')
        if not shortlisted_candidates:
            return "No detailed results available."
        
        details = "# 🏆 Shortlisted Candidates Details\n\n"
        
        for i, candidate in enumerate(shortlisted_candidates, 1):
            details += f"## {i}. {candidate.candidate_name}\n"
            details += f"**Overall Score**: {candidate.overall_score:.1f}/100\n"
            details += f"**Recommendation**: {candidate.recommendation}\n\n"
            
            if candidate.matched_skills:
                details += f"**✅ Matched Skills**: {', '.join(candidate.matched_skills)}\n"
            
            if candidate.missing_skills:
                details += f"**❌ Missing Skills**: {', '.join(candidate.missing_skills)}\n"
            
            details += "\n" + "-" * 50 + "\n\n"
        
        return details
    
    def create_interface(self):
        with gr.Blocks(title="HR Multi-Agent System") as interface:
            gr.Markdown("# 🤖 HR Multi-Agent Recruitment System")
            
            with gr.Row():
                with gr.Column():
                    title = gr.Textbox(label="Job Title", placeholder="e.g., Software Engineer")
                    company = gr.Textbox(label="Company Name", placeholder="e.g., TechCorp")
                    required_skills = gr.Textbox(label="Required Skills", placeholder="Python, JavaScript, React")
                    description = gr.Textbox(label="Job Description", lines=4)
                
                with gr.Column():
                    cv_files = gr.File(label="Upload CV Files", file_count="multiple")
            
            process_btn = gr.Button("🚀 Start Processing", variant="primary")
            
            with gr.Row():
                summary_output = gr.Markdown(label="Summary")
                details_output = gr.Markdown(label="Details")
            
            process_btn.click(
                fn=self.run_hr_workflow,
                inputs=[title, company, description, required_skills, cv_files],
                outputs=[summary_output, details_output]
            )
        
        return interface
    
    def launch(self):
        interface = self.create_interface()
        interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    app = GradioHRApp()
    app.launch() 