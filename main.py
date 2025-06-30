#!/usr/bin/env python3
"""
HR Multi-Agent System - Main Application
"""

import argparse
import os
import sys
from typing import List

from models import JobDescription
from workflow import HRWorkflow, get_state_value
from gradio_app import GradioHRApp

def create_sample_job_description() -> JobDescription:
    """Create a sample job description for demo purposes"""
    return JobDescription(
        title="Senior Software Engineer",
        company="TechCorp Solutions",
        location="San Francisco, CA",
        required_skills=[
            "Python", "JavaScript", "React", "Node.js", "SQL", 
            "Git", "REST APIs", "Agile Development"
        ],
        preferred_skills=[
            "AWS", "Docker", "Kubernetes", "GraphQL", "TypeScript",
            "Machine Learning", "DevOps", "Microservices"
        ],
        min_experience=3,
        education_requirements=["Bachelor's degree in Computer Science or related field"],
        job_type="Full-time",
        salary_range="$100,000 - $150,000",
        description="""
We are seeking a talented Senior Software Engineer to join our growing engineering team. 
You will be responsible for designing, developing, and maintaining scalable web applications 
and services that serve millions of users worldwide.

The ideal candidate has strong experience in full-stack development, with expertise in 
modern web technologies and cloud platforms. You should be passionate about writing 
clean, efficient code and collaborating with cross-functional teams to deliver 
high-quality software solutions.

This is an excellent opportunity to work with cutting-edge technologies, mentor junior 
developers, and contribute to the technical direction of our products.
        """.strip(),
        responsibilities=[
            "Design and develop scalable web applications",
            "Collaborate with product managers and designers",
            "Write clean, maintainable, and well-tested code",
            "Participate in code reviews and technical discussions",
            "Mentor junior developers and contribute to team growth",
            "Stay up-to-date with emerging technologies and best practices"
        ]
    )

def run_cli_demo(cv_directory: str):
    """Run a CLI demo of the HR workflow"""
    print("🚀 HR Multi-Agent System - CLI Demo")
    print("=" * 50)
    
    # Check if CV directory exists
    if not os.path.exists(cv_directory):
        print(f"❌ Error: CV directory '{cv_directory}' does not exist.")
        print("Please create the directory and add some CV files (PDF, DOCX, TXT)")
        return
    
    # Find CV files
    cv_files = []
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    for file in os.listdir(cv_directory):
        file_path = os.path.join(cv_directory, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file.lower())
            if ext in supported_extensions:
                cv_files.append(file_path)
    
    if not cv_files:
        print(f"❌ No CV files found in '{cv_directory}'")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"📄 Found {len(cv_files)} CV files:")
    for cv_file in cv_files:
        print(f"   - {os.path.basename(cv_file)}")
    
    # Create sample job description
    job_description = create_sample_job_description()
    print(f"\n📋 Job: {job_description.title} at {job_description.company}")
    
    # Run workflow
    workflow = HRWorkflow()
    try:
        results = workflow.run_workflow(job_description, cv_files)
        
        processing_status = get_state_value(results, 'processing_status')
        if processing_status == "completed":
            print("\n🎉 Workflow completed successfully!")
            print(f"📁 Results saved to: outputs/")
        else:
            print(f"\n❌ Workflow failed with status: {processing_status}")
            errors = get_state_value(results, 'errors')
            if errors:
                print("Errors:")
                for error in errors:
                    print(f"   - {error}")
    
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")

def create_sample_cvs():
    """Create sample CV files for demo purposes"""
    sample_cvs_dir = "sample_cvs"
    os.makedirs(sample_cvs_dir, exist_ok=True)
    
    sample_cvs = {
        "john_doe.txt": """
John Doe
Email: john.doe@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Senior Software Engineer with 5 years of experience in full-stack web development. 
Experienced in Python, JavaScript, React, and cloud technologies. Strong background 
in agile development and team collaboration.

TECHNICAL SKILLS
- Programming Languages: Python, JavaScript, TypeScript, Java
- Frontend: React, HTML5, CSS3, Redux
- Backend: Node.js, Django, Flask, Express.js
- Databases: PostgreSQL, MongoDB, Redis
- Cloud: AWS (EC2, S3, Lambda), Docker
- Tools: Git, Jenkins, Jira, VS Code

WORK EXPERIENCE
Senior Software Engineer | TechStart Inc. | 2021-2023
- Developed and maintained scalable web applications serving 100K+ users
- Led a team of 3 junior developers
- Implemented CI/CD pipelines reducing deployment time by 50%
- Built RESTful APIs and microservices architecture

Software Engineer | WebSolutions Co. | 2019-2021
- Developed responsive web applications using React and Node.js
- Collaborated with UX/UI designers to implement user interfaces
- Optimized database queries improving application performance by 30%

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2019

CERTIFICATIONS
- AWS Certified Solutions Architect
- Certified Scrum Master
        """.strip(),
        
        "jane_smith.txt": """
Jane Smith
Email: jane.smith@email.com  
Phone: (555) 987-6543
Location: New York, NY

PROFESSIONAL SUMMARY
Full-stack developer with 7 years of experience building enterprise applications. 
Expert in Python, React, and cloud infrastructure. Passionate about clean code 
and scalable architecture.

TECHNICAL SKILLS
- Languages: Python, JavaScript, Go, SQL
- Frontend: React, Vue.js, Angular, TypeScript
- Backend: Django, FastAPI, Node.js
- Databases: PostgreSQL, MySQL, MongoDB, ElasticSearch
- Cloud: AWS, Azure, Kubernetes, Docker
- DevOps: Jenkins, GitLab CI, Terraform

WORK EXPERIENCE
Lead Software Engineer | Enterprise Corp | 2020-2023
- Architected and built microservices handling 1M+ daily transactions
- Mentored team of 5 developers
- Implemented automated testing reducing bugs by 60%
- Designed scalable database schemas for high-traffic applications

Senior Developer | Innovation Labs | 2017-2020
- Built real-time data processing systems using Python and Apache Kafka
- Developed machine learning models for predictive analytics
- Created responsive web applications with React and Redux

EDUCATION
Master of Science in Computer Science
Stanford University | 2017

Bachelor of Engineering in Software Engineering  
MIT | 2015

CERTIFICATIONS
- AWS Certified DevOps Engineer
- Google Cloud Professional Developer
- Certified Kubernetes Administrator
        """.strip(),
        
        "mike_johnson.txt": """
Mike Johnson
Email: mike.johnson@email.com
Phone: (555) 456-7890
Location: Austin, TX

PROFESSIONAL SUMMARY
Frontend specialist with 4 years of experience in modern web development. 
Strong expertise in React, JavaScript, and user experience design. 
Passionate about creating intuitive and responsive user interfaces.

TECHNICAL SKILLS
- Frontend: React, JavaScript, HTML5, CSS3, SASS
- State Management: Redux, Context API, MobX
- Build Tools: Webpack, Vite, Parcel
- Testing: Jest, Cypress, React Testing Library
- Design: Figma, Adobe XD, Responsive Design
- Version Control: Git, GitHub, GitLab

WORK EXPERIENCE
Frontend Developer | UI Innovations | 2021-2023
- Developed responsive web applications using React and TypeScript
- Collaborated with designers to implement pixel-perfect UIs
- Optimized application performance achieving 95+ Lighthouse scores
- Built reusable component library used across multiple projects

Junior Frontend Developer | StartupXYZ | 2020-2021
- Created interactive user interfaces using React and JavaScript
- Implemented responsive designs for mobile and desktop
- Worked with REST APIs to integrate frontend with backend services

EDUCATION
Bachelor of Science in Web Development
University of Texas at Austin | 2020

PROJECTS
- E-commerce Platform: Built complete frontend using React, Redux, and Stripe integration
- Portfolio Website: Responsive personal website with modern design and animations
- Task Management App: React-based productivity app with drag-and-drop functionality
        """.strip()
    }
    
    for filename, content in sample_cvs.items():
        file_path = os.path.join(sample_cvs_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✅ Created sample CV files in '{sample_cvs_dir}/':")
    for filename in sample_cvs.keys():
        print(f"   - {filename}")
    
    return sample_cvs_dir

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="HR Multi-Agent Recruitment System")
    parser.add_argument(
        "--mode", 
        choices=["web", "cli", "demo"], 
        default="web",
        help="Run mode: web interface, CLI demo, or create demo files"
    )
    parser.add_argument(
        "--cv-dir", 
        default="sample_cvs",
        help="Directory containing CV files (for CLI mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        print("🚀 Starting HR Multi-Agent System Web Interface...")
        print("🌐 Open your browser to: http://localhost:7860")
        print("Press Ctrl+C to stop the server")
        
        app = GradioHRApp()
        app.launch()
    
    elif args.mode == "cli":
        run_cli_demo(args.cv_dir)
    
    elif args.mode == "demo":
        print("📝 Creating sample CV files for demo...")
        sample_dir = create_sample_cvs()
        print(f"\n🚀 Now run: python main.py --mode cli --cv-dir {sample_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 