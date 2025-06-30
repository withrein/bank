from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime, timedelta

from models import CandidateScore, JobDescription, EmailDraft, AgentState
from config import Config

class EmailAgent:
    """Agent responsible for drafting personalized emails to candidates"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.get_gemini_api_key(),
            temperature=Config.TEMPERATURE
        )
    
    def draft_interview_invitation(self, candidate: CandidateScore, 
                                 job_description: JobDescription) -> EmailDraft:
        """Draft an interview invitation email for a shortlisted candidate"""
        
        system_prompt = """You are an HR professional drafting interview invitation emails. Create a professional, warm, and personalized email invitation.

The email should:
1. Be professional yet friendly
2. Express enthusiasm about the candidate
3. Provide clear next steps
4. Include placeholder for interview details
5. Be personalized based on the candidate's background
6. Maintain a positive tone

Return the email content in a structured format with subject and body separated."""

        # Calculate suggested interview date (1 week from now)
        suggested_date = (datetime.now() + timedelta(days=7)).strftime("%A, %B %d, %Y")
        
        context = f"""
Draft an interview invitation email for:

CANDIDATE: {candidate.candidate_name}
POSITION: {job_description.title}
COMPANY: {job_description.company}
CANDIDATE SCORE: {candidate.overall_score:.1f}/100
CANDIDATE STRENGTHS: {', '.join(candidate.strengths[:3]) if candidate.strengths else 'Strong background'}

The email should invite them for an interview and mention their relevant qualifications.
Suggested interview date: {suggested_date}

Format the response as:
SUBJECT: [Email subject line]

BODY:
[Email body content]
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content)
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="interview_invitation",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company,
                interview_date=suggested_date,
                interview_time="[To be scheduled]"
            )
            
            return email_draft
            
        except Exception as e:
            print(f"Error drafting interview invitation for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "interview_invitation")
    
    def draft_rejection_email(self, candidate: CandidateScore, 
                            job_description: JobDescription) -> EmailDraft:
        """Draft a polite rejection email for a non-shortlisted candidate"""
        
        system_prompt = """You are an HR professional drafting rejection emails. Create a respectful, encouraging, and professional rejection email.

The email should:
1. Be respectful and empathetic
2. Thank the candidate for their interest
3. Provide constructive feedback if appropriate
4. Encourage future applications
5. Maintain the company's positive reputation
6. Be personalized but professional

Return the email content in a structured format with subject and body separated."""

        context = f"""
Draft a rejection email for:

CANDIDATE: {candidate.candidate_name}
POSITION: {job_description.title}
COMPANY: {job_description.company}
CANDIDATE SCORE: {candidate.overall_score:.1f}/100
REASON FOR REJECTION: {candidate.recommendation}

The email should politely inform them they were not selected but encourage future applications.

Format the response as:
SUBJECT: [Email subject line]

BODY:
[Email body content]
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content)
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="rejection",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company
            )
            
            return email_draft
            
        except Exception as e:
            print(f"Error drafting rejection email for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "rejection")
    
    def draft_follow_up_email(self, candidate: CandidateScore, 
                            job_description: JobDescription) -> EmailDraft:
        """Draft a follow-up email for candidates who need additional information"""
        
        system_prompt = """You are an HR professional drafting follow-up emails. Create a professional email requesting additional information from a candidate.

The email should:
1. Be professional and clear
2. Explain what additional information is needed
3. Provide clear instructions
4. Set expectations for timeline
5. Maintain enthusiasm about the candidate

Return the email content in a structured format with subject and body separated."""

        context = f"""
Draft a follow-up email for:

CANDIDATE: {candidate.candidate_name}
POSITION: {job_description.title}
COMPANY: {job_description.company}
MISSING INFORMATION: {', '.join(candidate.missing_skills[:3]) if candidate.missing_skills else 'Additional details needed'}

The email should request additional information or clarification about their background.

Format the response as:
SUBJECT: [Email subject line]

BODY:
[Email body content]
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content)
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="follow_up",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company
            )
            
            return email_draft
            
        except Exception as e:
            print(f"Error drafting follow-up email for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "follow_up")
    
    def _parse_email_content(self, email_content: str) -> tuple:
        """Parse email content to extract subject and body"""
        lines = email_content.split('\n')
        subject = ""
        body_lines = []
        
        in_body = False
        for line in lines:
            if line.strip().startswith('SUBJECT:'):
                subject = line.replace('SUBJECT:', '').strip()
            elif line.strip().startswith('BODY:'):
                in_body = True
            elif in_body:
                body_lines.append(line)
        
        if not subject:
            # Try to find subject in first line
            if lines and lines[0].strip():
                subject = lines[0].strip()
                body_lines = lines[1:]
        
        body = '\n'.join(body_lines).strip()
        
        # Fallback subjects if parsing failed
        if not subject:
            subject = "Regarding Your Application"
        
        if not body:
            body = "Thank you for your interest in our position."
        
        return subject, body
    
    def _get_candidate_email(self, candidate: CandidateScore) -> str:
        """Get candidate email (placeholder for now)"""
        # In a real system, this would extract email from the parsed CV
        return f"{candidate.candidate_name.lower().replace(' ', '.')}@email.com"
    
    def _create_fallback_email(self, candidate: CandidateScore, 
                             job_description: JobDescription, email_type: str) -> EmailDraft:
        """Create a fallback email when LLM generation fails"""
        
        fallback_emails = {
            "interview_invitation": {
                "subject": f"Interview Invitation - {job_description.title} Position",
                "body": f"Dear {candidate.candidate_name},\n\nThank you for your interest in the {job_description.title} position at {job_description.company}. We would like to invite you for an interview.\n\nWe will contact you shortly to schedule a convenient time.\n\nBest regards,\nHR Team"
            },
            "rejection": {
                "subject": f"Update on Your Application - {job_description.title} Position",
                "body": f"Dear {candidate.candidate_name},\n\nThank you for your interest in the {job_description.title} position at {job_description.company}. After careful consideration, we have decided to move forward with other candidates.\n\nWe encourage you to apply for future opportunities.\n\nBest regards,\nHR Team"
            },
            "follow_up": {
                "subject": f"Additional Information Needed - {job_description.title} Position",
                "body": f"Dear {candidate.candidate_name},\n\nThank you for your application for the {job_description.title} position. We would like to request some additional information to complete our review.\n\nPlease respond at your earliest convenience.\n\nBest regards,\nHR Team"
            }
        }
        
        template = fallback_emails.get(email_type, fallback_emails["follow_up"])
        
        return EmailDraft(
            recipient_name=candidate.candidate_name,
            recipient_email=self._get_candidate_email(candidate),
            email_type=email_type,
            subject=template["subject"],
            body=template["body"],
            job_title=job_description.title,
            company_name=job_description.company
        )
    
    def draft_emails_for_all_candidates(self, all_candidates: List[CandidateScore],
                                      shortlisted_candidates: List[CandidateScore],
                                      job_description: JobDescription) -> List[EmailDraft]:
        """Draft appropriate emails for all candidates"""
        email_drafts = []
        
        shortlisted_names = {candidate.candidate_name for candidate in shortlisted_candidates}
        
        # Draft interview invitations for shortlisted candidates
        for candidate in shortlisted_candidates:
            print(f"Drafting interview invitation for: {candidate.candidate_name}")
            email = self.draft_interview_invitation(candidate, job_description)
            email_drafts.append(email)
        
        # Draft rejection emails for non-shortlisted candidates
        for candidate in all_candidates:
            if candidate.candidate_name not in shortlisted_names:
                print(f"Drafting rejection email for: {candidate.candidate_name}")
                email = self.draft_rejection_email(candidate, job_description)
                email_drafts.append(email)
        
        return email_drafts
    
    def process(self, state: AgentState) -> AgentState:
        """Process email drafting in the agent state"""
        if not state.candidate_scores:
            state.errors.append("No candidate scores available for email drafting")
            return state
        
        if not state.job_description:
            state.errors.append("No job description available for email drafting")
            return state
        
        try:
            print("📧 Email Agent: Drafting candidate emails...")
            state.current_step = "drafting_emails"
            
            # Draft emails for all candidates
            email_drafts = self.draft_emails_for_all_candidates(
                state.candidate_scores,
                state.shortlisted_candidates or [],
                state.job_description
            )
            state.email_drafts = email_drafts
            
            print(f"✅ Email Agent: Drafted {len(email_drafts)} emails")
            
            # Log email drafting results
            email_types = {}
            for email in email_drafts:
                email_type = email.email_type
                email_types[email_type] = email_types.get(email_type, 0) + 1
            
            for email_type, count in email_types.items():
                print(f"   - {email_type}: {count} emails")
            
            state.current_step = "emails_drafted"
            
        except Exception as e:
            error_msg = f"Email Agent error: {str(e)}"
            print(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 