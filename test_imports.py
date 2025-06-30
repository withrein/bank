#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

def test_imports():
    print("Testing imports...")
    
    try:
        print("✓ Testing pydantic...")
        from pydantic import BaseModel, Field, ConfigDict
        
        print("✓ Testing langchain...")
        from langchain.schema import HumanMessage, SystemMessage
        
        print("✓ Testing langgraph...")
        from langgraph.graph import StateGraph, END
        
        print("✓ Testing langchain-google-genai...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        print("✓ Testing gradio...")
        import gradio as gr
        
        print("✓ Testing file processing...")
        import PyPDF2
        from docx import Document
        import pdfplumber
        
        print("✓ Testing utilities...")
        import pandas as pd
        import numpy as np
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 Ready to run the HR Multi-Agent System!")
    else:
        print("\n❌ Please fix the import issues before proceeding.")
