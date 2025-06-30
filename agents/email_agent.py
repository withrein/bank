import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime, timedelta

from models import CandidateScore, JobDescription, EmailDraft, AgentState
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailAgent:
    """Enhanced Email Agent with bilingual support for drafting personalized emails"""
    
    def __init__(self):
        model_config = Config.get_current_model_config()
        self.llm = ChatOpenAI(
            model=model_config["model"],
            openai_api_key=model_config["api_key"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )
        self.email_templates = Config.EMAIL_LANGUAGES
        self.mongolian_keywords = Config.get_language_keywords("mn")
        self.english_keywords = Config.get_language_keywords("en")
    
    def detect_language_preference(self, candidate: CandidateScore, job_description: JobDescription) -> str:
        """Detect preferred language for email communication"""
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
    
    def draft_interview_invitation(self, candidate: CandidateScore, 
                                 job_description: JobDescription) -> EmailDraft:
        """Draft an interview invitation email with bilingual support"""
        
        language = self.detect_language_preference(candidate, job_description)
        
        if language == "mn":
            system_prompt = """Та HR мэргэжилтэн бөгөөд ярилцлагад урих имэйл бичиж байна. Мэргэжлийн, дулаахан, хувийн шинж чанартай имэйл үүсгэнэ үү.

Имэйл нь:
1. Мэргэжлийн боловч найрсаг байх
2. Ажилтны талаар баяр хөөрийг илэрхийлэх
3. Дараагийн алхмуудыг тодорхой заах
4. Ярилцлагын дэлгэрэнгүй мэдээллийн зай үлдээх
5. Ажилтны мэдлэг туршлагад тулгуурлан хувийн шинж чанартай байх
6. Эерэг хэв маягтай байх

Имэйлийн гарчиг болон мазмуныг тусад нь бичнэ үү."""
            
            context = f"""
Дараах ажилтанд ярилцлагад урих имэйл бичнэ үү:

АЖИЛТАН: {candidate.candidate_name}
АЛБАН ТУШААЛ: {job_description.title}
КОМПАНИ: {job_description.company}
АЖИЛТНЫ ОНОО: {candidate.overall_score:.1f}/100
АЖИЛТНЫ ДАВУУ ТАЛ: {', '.join(candidate.strengths[:3]) if candidate.strengths else 'Мэдлэг туршлага сайн'}

Ярилцлагад урих имэйл бичиж, холбогдох мэргэшлийг дурдана уу.

Дараах хэлбэрээр бичнэ үү:
ГАРЧИГ: [Имэйлийн гарчиг]

АГУУЛГА:
[Имэйлийн агуулга]
"""
        else:
            system_prompt = """You are an HR professional drafting interview invitation emails. Create a professional, warm, and personalized email invitation.

The email should:
1. Be professional yet friendly
2. Express enthusiasm about the candidate
3. Provide clear next steps
4. Include placeholder for interview details
5. Be personalized based on the candidate's background
6. Maintain a positive tone

Return the email content in a structured format with subject and body separated."""
            
            context = f"""
Draft an interview invitation email for:

CANDIDATE: {candidate.candidate_name}
POSITION: {job_description.title}
COMPANY: {job_description.company}
CANDIDATE SCORE: {candidate.overall_score:.1f}/100
CANDIDATE STRENGTHS: {', '.join(candidate.strengths[:3]) if candidate.strengths else 'Strong background'}

The email should invite them for an interview and mention their relevant qualifications.

Format the response as:
SUBJECT: [Email subject line]

BODY:
[Email body content]
"""

        # Calculate suggested interview date (1 week from now)
        suggested_date = (datetime.now() + timedelta(days=7)).strftime("%A, %B %d, %Y")

        try:
            logger.info(f"📧 Drafting interview invitation for {candidate.candidate_name}")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content, language)
            
            # Use template subject if parsing fails
            if not subject:
                template_key = "interview_invitation_subject"
                subject = self.email_templates[language][template_key].format(
                    company=job_description.company,
                    job_title=job_description.title
                )
            
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
            
            logger.info(f"✅ Created interview invitation for {candidate.candidate_name}")
            return email_draft
            
        except Exception as e:
            logger.error(f"❌ Error drafting interview invitation for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "interview_invitation", language)
    
    def draft_rejection_email(self, candidate: CandidateScore, 
                            job_description: JobDescription) -> EmailDraft:
        """Draft a polite rejection email with bilingual support"""
        
        language = self.detect_language_preference(candidate, job_description)
        
        if language == "mn":
            system_prompt = """Та HR мэргэжилтэн бөгөөд татгалзах имэйл бичиж байна. Хүндэтгэсэн, урамшуулсан, мэргэжлийн татгалзах имэйл үүсгэнэ үү.

Имэйл нь:
1. Хүндэтгэсэн, өрөвдмөөр байх
2. Ажилтны сонирхлыг талархах
3. Тохиромжтой бол бүтээлийн санал хүргэх
4. Ирээдүйн хүсэлтийг урамшуулах
5. Компанийн эерэг нэр хүндийг хадгалах
6. Хувийн шинж чанартай боловч мэргэжлийн байх

Имэйлийн гарчиг болон мазмуныг тусад нь бичнэ үү."""
            
            context = f"""
Дараах ажилтанд татгалзах имэйл бичнэ үү:

АЖИЛТАН: {candidate.candidate_name}
АЛБАН ТУШААЛ: {job_description.title}
КОМПАНИ: {job_description.company}
АЖИЛТНЫ ОНОО: {candidate.overall_score:.1f}/100
ТАТГАЛЗАХ ШАЛТГААН: {candidate.recommendation}

Тэднийг сонгогдоогүй гэдгийг эелдэгээр мэдэгдэж, ирээдүйн хүсэлтийг урамшуулах имэйл бичнэ үү.

Дараах хэлбэрээр бичнэ үү:
ГАРЧИГ: [Имэйлийн гарчиг]

АГУУЛГА:
[Имэйлийн агуулга]
"""
        else:
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
            logger.info(f"📧 Drafting rejection email for {candidate.candidate_name}")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content, language)
            
            # Use template subject if parsing fails
            if not subject:
                template_key = "rejection_subject"
                subject = self.email_templates[language][template_key].format(
                    company=job_description.company
                )
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="rejection",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company
            )
            
            logger.info(f"✅ Created rejection email for {candidate.candidate_name}")
            return email_draft
            
        except Exception as e:
            logger.error(f"❌ Error drafting rejection email for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "rejection", language)
    
    def draft_follow_up_email(self, candidate: CandidateScore, 
                            job_description: JobDescription) -> EmailDraft:
        """Draft a follow-up email with bilingual support"""
        
        language = self.detect_language_preference(candidate, job_description)
        
        if language == "mn":
            system_prompt = """Та HR мэргэжилтэн бөгөөд дагалдах имэйл бичиж байна. Ажилтнаас нэмэлт мэдээлэл хүсэх мэргэжлийн имэйл үүсгэнэ үү.

Имэйл нь:
1. Мэргэжлийн, тодорхой байх
2. Ямар нэмэлт мэдээлэл хэрэгтэйг тайлбарлах
3. Тодорхой заавар өгөх
4. Хугацааны хүлээлтийг тогтоох
5. Ажилтны талаар сэтгэл хөдлөлийг хадгалах

Имэйлийн гарчиг болон мазмуныг тусад нь бичнэ үү."""
            
            context = f"""
Дараах ажилтанд дагалдах имэйл бичнэ үү:

АЖИЛТАН: {candidate.candidate_name}
АЛБАН ТУШААЛ: {job_description.title}
КОМПАНИ: {job_description.company}
ДУТУУ МЭДЭЭЛЭЛ: {', '.join(candidate.missing_skills[:3]) if candidate.missing_skills else 'Нэмэлт дэлгэрэнгүй мэдээлэл хэрэгтэй'}

Тэдний мэдлэг туршлагын талаар нэмэлт мэдээлэл эсвэл тодруулга хүсэх имэйл бичнэ үү.

Дараах хэлбэрээр бичнэ үү:
ГАРЧИГ: [Имэйлийн гарчиг]

АГУУЛГА:
[Имэйлийн агуулга]
"""
        else:
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
            logger.info(f"📧 Drafting follow-up email for {candidate.candidate_name}")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content, language)
            
            # Use default subject if parsing fails
            if not subject:
                subject = f"Additional Information Required - {job_description.title}" if language == "en" else f"Нэмэлт мэдээлэл хэрэгтэй - {job_description.title}"
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="follow_up",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company
            )
            
            logger.info(f"✅ Created follow-up email for {candidate.candidate_name}")
            return email_draft
            
        except Exception as e:
            logger.error(f"❌ Error drafting follow-up email for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "follow_up", language)
    
    def draft_acknowledgment_email(self, candidate: CandidateScore, 
                                 job_description: JobDescription) -> EmailDraft:
        """Draft an application acknowledgment email"""
        
        language = self.detect_language_preference(candidate, job_description)
        
        if language == "mn":
            system_prompt = """Та HR мэргэжилтэн бөгөөд өргөдлийг хүлээн авсан талаар баталгаажуулах имэйл бичиж байна.

Имэйл нь:
1. Өргөдлийг хүлээн авсныг баталгаажуулах
2. Дараагийн алхмуудыг тайлбарлах
3. Хүлээх хугацааг заах
4. Талархал илэрхийлэх
5. Богино боловч мэдээллийн байх

Имэйлийн гарчиг болон мазмуныг тусад нь бичнэ үү."""
            
            context = f"""
АЖИЛТАН: {candidate.candidate_name}
АЛБАН ТУШААЛ: {job_description.title}
КОМПАНИ: {job_description.company}

Өргөдлийг хүлээн авсан талаар баталгаажуулах имэйл бичнэ үү.

Дараах хэлбэрээр бичнэ үү:
ГАРЧИГ: [Имэйлийн гарчиг]

АГУУЛГА:
[Имэйлийн агуулга]
"""
        else:
            system_prompt = """You are an HR professional drafting application acknowledgment emails.

The email should:
1. Confirm receipt of application
2. Explain next steps
3. Set timeline expectations
4. Express appreciation
5. Be brief but informative

Return the email content in a structured format with subject and body separated."""
            
            context = f"""
CANDIDATE: {candidate.candidate_name}
POSITION: {job_description.title}
COMPANY: {job_description.company}

Draft an acknowledgment email confirming receipt of their application.

Format the response as:
SUBJECT: [Email subject line]

BODY:
[Email body content]
"""

        try:
            logger.info(f"📧 Drafting acknowledgment email for {candidate.candidate_name}")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            email_content = response.content.strip()
            
            # Parse subject and body
            subject, body = self._parse_email_content(email_content, language)
            
            # Use template subject if parsing fails
            if not subject:
                template_key = "acknowledgment_subject"
                subject = self.email_templates[language][template_key].format(
                    company=job_description.company
                )
            
            email_draft = EmailDraft(
                recipient_name=candidate.candidate_name,
                recipient_email=self._get_candidate_email(candidate),
                email_type="acknowledgment",
                subject=subject,
                body=body,
                job_title=job_description.title,
                company_name=job_description.company
            )
            
            logger.info(f"✅ Created acknowledgment email for {candidate.candidate_name}")
            return email_draft
            
        except Exception as e:
            logger.error(f"❌ Error drafting acknowledgment email for {candidate.candidate_name}: {str(e)}")
            return self._create_fallback_email(candidate, job_description, "acknowledgment", language)
    
    def _parse_email_content(self, email_content: str, language: str = "en") -> tuple:
        """Parse email content to extract subject and body with improved parsing"""
        try:
            # Look for subject patterns in both languages
            subject_patterns = [
                "SUBJECT:", "Subject:", "ГАРЧИГ:", "Гарчиг:",
                "СЭДЭВ:", "Сэдэв:", "TITLE:", "Title:"
            ]
            
            body_patterns = [
                "BODY:", "Body:", "АГУУЛГА:", "Агуулга:",
                "БИЕТ:", "Биет:", "CONTENT:", "Content:"
            ]
            
            lines = email_content.split('\n')
            subject = ""
            body = ""
            body_started = False
            
            for line in lines:
                line = line.strip()
                
                # Check for subject line
                if not subject and any(pattern in line for pattern in subject_patterns):
                    subject = line.split(':', 1)[1].strip() if ':' in line else ""
                    continue
                
                # Check for body start
                if any(pattern in line for pattern in body_patterns):
                    body_started = True
                    body_content = line.split(':', 1)[1].strip() if ':' in line else ""
                    if body_content:
                        body = body_content
                    continue
                
                # Add to body if we've started
                if body_started and line:
                    if body:
                        body += "\n" + line
                    else:
                        body = line
                
                # If no explicit markers found, treat first line as subject
                if not subject and not body_started and line:
                    subject = line
                    body_started = True
                    continue
            
            # Fallback: if we have content but no clear separation
            if not subject and not body and email_content.strip():
                lines = email_content.strip().split('\n')
                if lines:
                    subject = lines[0].strip()
                    if len(lines) > 1:
                        body = '\n'.join(lines[1:]).strip()
            
            return subject.strip(), body.strip()
            
        except Exception as e:
            logger.error(f"Error parsing email content: {str(e)}")
            return "", email_content.strip()
    
    def _get_candidate_email(self, candidate: CandidateScore) -> str:
        """Extract candidate email or generate placeholder"""
        # This would typically extract from candidate data
        # For now, return a placeholder
        return f"{candidate.candidate_name.lower().replace(' ', '.')}@example.com"
    
    def _create_fallback_email(self, candidate: CandidateScore, 
                             job_description: JobDescription, email_type: str, language: str = "en") -> EmailDraft:
        """Create a fallback email when LLM generation fails"""
        
        fallback_templates = {
            "en": {
                "interview_invitation": {
                    "subject": f"Interview Invitation - {job_description.title}",
                    "body": f"Dear {candidate.candidate_name},\n\nWe are pleased to invite you for an interview for the {job_description.title} position at {job_description.company}.\n\nWe will contact you soon with interview details.\n\nBest regards,\nHR Team"
                },
                "rejection": {
                    "subject": f"Application Update - {job_description.title}",
                    "body": f"Dear {candidate.candidate_name},\n\nThank you for your interest in the {job_description.title} position at {job_description.company}.\n\nAfter careful consideration, we have decided to move forward with other candidates.\n\nWe encourage you to apply for future opportunities.\n\nBest regards,\nHR Team"
                },
                "follow_up": {
                    "subject": f"Additional Information Required - {job_description.title}",
                    "body": f"Dear {candidate.candidate_name},\n\nThank you for your application for the {job_description.title} position.\n\nWe need some additional information to complete our review.\n\nPlease contact us at your earliest convenience.\n\nBest regards,\nHR Team"
                },
                "acknowledgment": {
                    "subject": f"Application Received - {job_description.title}",
                    "body": f"Dear {candidate.candidate_name},\n\nWe have received your application for the {job_description.title} position at {job_description.company}.\n\nWe will review your application and contact you with updates.\n\nThank you for your interest.\n\nBest regards,\nHR Team"
                }
            },
            "mn": {
                "interview_invitation": {
                    "subject": f"{job_description.company}-д {job_description.title} албан тушаалд ярилцлагад урих",
                    "body": f"Эрхэм {candidate.candidate_name},\n\n{job_description.company} компанийн {job_description.title} албан тушаалд ярилцлагад урьж байна.\n\nЯрилцлагын дэлгэрэнгүй мэдээллийг удахгүй хүргэх болно.\n\nХүндэтгэсэн,\nХүний нөөцийн хэлтэс"
                },
                "rejection": {
                    "subject": f"{job_description.company}-ээс хариу",
                    "body": f"Эрхэм {candidate.candidate_name},\n\n{job_description.company} компанийн {job_description.title} албан тушаалд сонирхол танилцуулсанд талархаж байна.\n\nСайтар судалсны үндсэн дээр бид бусад ажилтнуудтай үргэлжлүүлэх шийдвэр гаргалаа.\n\nИрээдүйд гарах боломжуудад хүсэлт гаргахыг урьж байна.\n\nХүндэтгэсэн,\nХүний нөөцийн хэлтэс"
                },
                "follow_up": {
                    "subject": f"Нэмэлт мэдээлэл хэрэгтэй - {job_description.title}",
                    "body": f"Эрхэм {candidate.candidate_name},\n\n{job_description.title} албан тушаалд хүсэлт гаргасанд талархаж байна.\n\nТаны хүсэлтийг бүрэн хянахын тулд нэмэлт мэдээлэл хэрэгтэй байна.\n\nБоломжийн хугацаандаа холбогдоно уу.\n\nХүндэтгэсэн,\nХүний нөөцийн хэлтэс"
                },
                "acknowledgment": {
                    "subject": f"Таны өргөдлийг хүлээн авлаа - {job_description.title}",
                    "body": f"Эрхэм {candidate.candidate_name},\n\n{job_description.company} компанийн {job_description.title} албан тушаалд таны өргөдлийг хүлээн авлаа.\n\nТаны өргөдлийг хянаж, мэдээлэл хүргэх болно.\n\nСонирхол танилцуулсанд талархаж байна.\n\nХүндэтгэсэн,\nХүний нөөцийн хэлтэс"
                }
            }
        }
        
        template = fallback_templates.get(language, fallback_templates["en"]).get(email_type, fallback_templates["en"]["acknowledgment"])
        
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
        """Draft emails for all candidates with enhanced categorization"""
        email_drafts = []
        shortlisted_names = {candidate.candidate_name for candidate in shortlisted_candidates}
        
        logger.info(f"📧 Drafting emails for {len(all_candidates)} candidates")
        logger.info(f"   - {len(shortlisted_candidates)} shortlisted for interviews")
        logger.info(f"   - {len(all_candidates) - len(shortlisted_candidates)} to be rejected")
        
        # Draft interview invitations for shortlisted candidates
        for candidate in shortlisted_candidates:
            logger.info(f"📧 Interview invitation: {candidate.candidate_name}")
            email = self.draft_interview_invitation(candidate, job_description)
            email_drafts.append(email)
        
        # Draft rejection emails for non-shortlisted candidates
        for candidate in all_candidates:
            if candidate.candidate_name not in shortlisted_names:
                logger.info(f"📧 Rejection email: {candidate.candidate_name}")
                email = self.draft_rejection_email(candidate, job_description)
                email_drafts.append(email)
        
        # Draft acknowledgment emails for all candidates
        for candidate in all_candidates:
            logger.info(f"📧 Acknowledgment email: {candidate.candidate_name}")
            email = self.draft_acknowledgment_email(candidate, job_description)
            email_drafts.append(email)
        
        logger.info(f"✅ Completed drafting {len(email_drafts)} emails")
        
        # Log email type summary
        email_types = {}
        for email in email_drafts:
            email_type = email.email_type
            email_types[email_type] = email_types.get(email_type, 0) + 1
        
        for email_type, count in email_types.items():
            logger.info(f"   - {email_type.replace('_', ' ').title()}: {count}")
        
        return email_drafts
    
    def process(self, state: AgentState) -> AgentState:
        """Process email drafting in the agent state"""
        if not state.candidate_scores:
            state.errors.append("No candidate scores available for email drafting")
            return state
        
        if not state.shortlisted_candidates:
            state.errors.append("No shortlisted candidates available for email drafting")
            return state
        
        if not state.job_description:
            state.errors.append("No job description available for email drafting")
            return state
        
        try:
            logger.info("📧 Email Agent: Starting email drafting...")
            state.current_step = "drafting_emails"
            
            # Draft emails for all candidates
            email_drafts = self.draft_emails_for_all_candidates(
                state.candidate_scores,
                state.shortlisted_candidates,
                state.job_description
            )
            state.email_drafts = email_drafts
            
            logger.info(f"✅ Email Agent: Successfully drafted {len(email_drafts)} emails")
            
            # Log email drafting results
            email_types = {}
            for email in email_drafts:
                email_type = email.email_type
                email_types[email_type] = email_types.get(email_type, 0) + 1
            
            for email_type, count in email_types.items():
                logger.info(f"   - {email_type.replace('_', ' ').title()}: {count}")
            
            state.current_step = "emails_drafted"
            
        except Exception as e:
            error_msg = f"Email Agent error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            state.errors.append(error_msg)
        
        return state 