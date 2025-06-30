# 🤖 HR Multi-Agent Recruitment System

An intelligent multi-agent system that automates the entire HR recruitment workflow using AI-powered agents. Built with **LangGraph**, **Gemini API**, and **Gradio** for seamless candidate screening, evaluation, and communication.

## 🎯 Overview

This system automates the complete HR recruitment process from CV parsing to email drafting, using specialized AI agents that work together to:

- **Parse CVs** and extract structured candidate information
- **Score candidates** against job requirements with detailed analysis
- **Shortlist top candidates** based on configurable criteria
- **Generate tailored interview questions** for each candidate
- **Draft personalized emails** for invitations and rejections

## 🏗️ Architecture

The system uses a **multi-agent architecture** with specialized agents orchestrated by **LangGraph**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CV Parser     │───▶│  Scoring Agent  │───▶│ Shortlisting    │
│     Agent       │    │                 │    │     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Email Agent   │◀───│  Interview      │◀────────────┘
│                 │    │     Agent       │
└─────────────────┘    └─────────────────┘
```

### 🧩 Agent Breakdown

1. **CV Parsing Agent**
   - Extracts text from PDF, DOCX, DOC, TXT files
   - Uses OCR and advanced parsing techniques
   - Combines LLM analysis with regex patterns
   - Outputs structured candidate data

2. **JD Matching & Scoring Agent**
   - Compares CVs against job requirements
   - Multi-dimensional scoring (skills, experience, education)
   - Provides detailed analysis and recommendations
   - Outputs JSON-formatted scores (0-100)

3. **Shortlisting Agent**
   - Ranks candidates by overall score
   - Applies configurable thresholds
   - Selects top N candidates for review
   - Generates summary reports

4. **Interview Question Agent**
   - Creates tailored questions for each candidate
   - Generates technical, behavioral, and role-specific questions
   - Considers candidate background and job requirements
   - Provides expected answer points

5. **Email Draft Agent**
   - Composes personalized emails
   - Handles invitations, rejections, and follow-ups
   - Includes dynamic variables (name, role, date)
   - Maintains professional tone

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Gemini API key
- Required dependencies (see `requirements.txt`)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd hr-multi-agent-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**
Update the API keys in `config.py`:
```python
GEMINI_API_KEY = "your-gemini-api-key"
```

### Running the System

#### 🌐 Web Interface (Recommended)
```bash
python main.py --mode web
```
Open your browser to `http://localhost:7860`

#### 🖥️ CLI Demo
```bash
# Create sample CV files
python main.py --mode demo

# Run CLI demo
python main.py --mode cli --cv-dir sample_cvs
```

#### 📊 Direct Usage
```python
from workflow import HRWorkflow
from models import JobDescription

# Create job description
job_desc = JobDescription(
    title="Software Engineer",
    company="TechCorp",
    description="...",
    required_skills=["Python", "JavaScript"]
)

# Run workflow
workflow = HRWorkflow()
results = workflow.run_workflow(job_desc, ["cv1.pdf", "cv2.docx"])
```

## 📋 Features

### ✅ CV Processing
- **Multi-format support**: PDF, DOCX, DOC, TXT
- **Intelligent extraction**: Contact info, skills, experience, education
- **Error handling**: Graceful handling of corrupted/unreadable files
- **Batch processing**: Handle multiple CVs simultaneously

### 📊 Candidate Scoring
- **Multi-dimensional analysis**: Skills, experience, education, cultural fit
- **Weighted scoring**: Configurable weights for different criteria
- **Detailed breakdown**: Individual scores and overall ranking
- **LLM-powered insights**: Strengths, weaknesses, and recommendations

### 🎯 Smart Shortlisting
- **Configurable thresholds**: Minimum score requirements
- **Flexible limits**: Adjustable number of candidates to shortlist
- **Ranking algorithms**: Multiple sorting and filtering options
- **Summary reports**: Clear overview of selection criteria

### ❓ Interview Questions
- **Tailored questions**: Customized for each candidate's background
- **Multiple categories**: Technical, behavioral, role-specific
- **Difficulty levels**: Easy, medium, hard questions
- **Answer guidance**: Expected response points for interviewers

### 📧 Email Automation
- **Template variety**: Invitations, rejections, follow-ups
- **Personalization**: Dynamic content based on candidate data
- **Professional tone**: Consistent brand voice and messaging
- **Bulk generation**: Efficient processing of multiple emails

## 🛠️ Configuration

### Application Settings (`config.py`)
```python
# Shortlisting criteria
MAX_CANDIDATES_TO_SHORTLIST = 5
MINIMUM_SCORE_THRESHOLD = 60

# File processing
ALLOWED_CV_FORMATS = ['.pdf', '.docx', '.doc', '.txt']
MAX_FILE_SIZE_MB = 10

# LLM settings
GEMINI_MODEL = "gemini-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
```

### Customization Options
- **Scoring weights**: Adjust importance of skills vs experience
- **Question types**: Add custom question categories
- **Email templates**: Modify email styles and content
- **Skills database**: Extend technical skills recognition

## 📁 Project Structure

```
hr-multi-agent-system/
├── agents/                 # Specialized AI agents
│   ├── cv_parser_agent.py
│   ├── scoring_agent.py
│   ├── shortlisting_agent.py
│   ├── interview_agent.py
│   └── email_agent.py
├── models.py              # Pydantic data models
├── workflow.py            # LangGraph workflow orchestrator
├── utils.py               # Utility functions
├── config.py              # Configuration settings
├── gradio_app.py          # Web interface
├── main.py                # Application entry point
├── requirements.txt       # Dependencies
└── outputs/               # Generated results
    ├── parsed_cvs.json
    ├── candidate_scores.json
    ├── shortlisted_candidates.json
    ├── interview_questions.json
    └── email_drafts.json
```

## 📊 Output Examples

### Candidate Scoring
```json
{
  "candidate_name": "John Doe",
  "overall_score": 87.5,
  "skills_match_score": 90.0,
  "experience_score": 85.0,
  "education_score": 80.0,
  "matched_skills": ["Python", "JavaScript", "React"],
  "missing_skills": ["Kubernetes", "GraphQL"],
  "recommendation": "Highly Recommended - Strong candidate with excellent fit"
}
```

### Interview Questions
```json
{
  "candidate_name": "John Doe",
  "technical_questions": [
    {
      "question": "How would you optimize a React application for performance?",
      "category": "technical",
      "difficulty": "medium",
      "expected_answer_points": ["Code splitting", "Memoization", "Bundle optimization"]
    }
  ]
}
```

### Email Drafts
```json
{
  "recipient_name": "John Doe",
  "email_type": "interview_invitation",
  "subject": "Interview Invitation - Software Engineer Position",
  "body": "Dear John,\n\nWe were impressed by your background...",
  "interview_date": "Friday, December 15, 2023"
}
```

## 🔧 Advanced Usage

### Custom Agents
Extend the system by creating custom agents:

```python
from models import AgentState

class CustomAgent:
    def process(self, state: AgentState) -> AgentState:
        # Your custom logic here
        return state
```

### Workflow Customization
Modify the workflow graph in `workflow.py`:

```python
# Add custom nodes
workflow.add_node("custom_step", self._custom_node)
workflow.add_edge("shortlist_candidates", "custom_step")
```

### Integration APIs
Use the system programmatically:

```python
from workflow import HRWorkflow

workflow = HRWorkflow()
status = workflow.get_workflow_status()
print(f"Current step: {status['current_step']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for workflow orchestration
- **Google Gemini** for LLM capabilities
- **Gradio** for the web interface
- **Pydantic** for data validation
- **PyPDF2** and **pdfplumber** for PDF processing

## 📞 Support

For questions, issues, or contributions:

- 📧 Email: support@hr-agents.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Built with ❤️ for HR professionals and developers** 