AI-Powered Market Analysis & Content Generation Tool

Overview

This Streamlit-based application provides AI-powered market analysis and content generation. It allows users to upload quarterly reports in PDF format, extract insights, predict trends, and generate marketing content and images based on text inputs.

Features

Market Analysis

Extracts text from uploaded PDFs

Summarizes content using BART

Extracts named entities with spaCy

Predicts market trends using Gemini API

Provides a cumulative analysis of all uploaded reports

Supports question-answering on extracted summaries

Downloadable cumulative analysis in JSON format

Marketing Image Generation

Generates images based on textual descriptions using Unsplash API

Marketing Content Generation

Generates marketing copy using Gemini API based on user-provided prompts

Installation

Clone this repository:

git clone https://github.com/your-repo/ai-market-analysis.git
cd ai-market-analysis

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Dependencies

streamlit

PyPDF2

torch

spacy

google.generativeai

transformers

PIL (Pillow)

requests

Usage

Market Analysis: Upload PDFs via the sidebar, and the tool will extract insights.

Image Generation: Input a description, and the tool will fetch a relevant image.

Text Generation: Provide a prompt, and the tool will generate marketing content.

API Keys

Make sure to set your Gemini API key before running the application:

genai.configure(api_key="YOUR_GEMINI_API_KEY")

Future Enhancements

Improve summarization handling for longer texts

Optimize model loading for performance

Add more robust error handling for missing or incomplete data

License

This project is open-source and available under the MIT License.

