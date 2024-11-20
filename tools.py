import re
import requests
from typing import Dict
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import StructuredTool
from crewai import LLM
from pydantic import BaseModel, Field

class HumanizeInput(BaseModel):
    query: str 

class SOPGenerationTools:
    """Tools for SOP generation, web scraping, and text transformation"""

    def __init__(self, llm_config: Dict[str, str]):
        self.search = DuckDuckGoSearchRun()
        self.llm = LLM(
            model=llm_config.get('model'),
            api_key=llm_config.get('api_key'),
        )
        
        # Contractions and idioms
        self.contractions = {
            'cannot': "can't",
            'will not': "won't",
            'shall not': "shan't",
            'do not': "don't",
            'does not': "doesn't",
            'did not': "didn't",
            'is not': "isn't",
            'are not': "aren't",
            'was not': "wasn't",
            'have not': "haven't",
            'has not': "hasn't",
            'I am': "I'm",
            'you are': "you're",
            'they are': "they're"
        }
        self.idioms = {
            'in addition': "on top of that",
            'for example': "like",
            'therefore': "so",
            'however': "though",
            'nevertheless': "still",
            'subsequently': "then",
            'furthermore': "also",
            'in conclusion': "to wrap things up"
        }

    def search_university(self, query: str) -> str:
        """Search for university information using DuckDuckGo"""
        try:
            results = self.search.run(query)
            return results
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def scrape_webpage(self, url: str) -> str:
        """Scrape content from a webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            return f"Error scraping webpage: {str(e)}"

    def generate_sop(self, student_info: Dict, program_info: Dict) -> str:
        """Generate SOP using LLM"""
        prompt = f"""Generate a Statement of Purpose for a student with the following details:

Student Background:
{student_info}

Target Program:
{program_info}

Write a compelling SOP that:
1. Shows genuine enthusiasm for the field
2. Connects past experiences with future goals
3. Demonstrates fit with the program
4. Maintains a personal and authentic voice
5. Includes specific details about the university and program

Format: Write in first person, approximately 1000 words.
"""
        try:
            response = self.llm.complete(prompt)
            return response
        except Exception as e:
            return f"Error generating SOP: {str(e)}"

    def humanize_text(self, text: str) -> str:
        """Make text more conversational by applying contractions and idioms"""
        text = self.apply_contractions(text)
        return self.apply_idioms(text)

    def apply_contractions(self, text: str) -> str:
        """Apply contractions to the text"""
        for formal, contraction in self.contractions.items():
            pattern = r'\b' + formal + r'\b'
            text = re.sub(pattern, contraction, text, flags=re.IGNORECASE)
        return text

    def apply_idioms(self, text: str) -> str:
        """Replace formal phrases with more casual idioms"""
        for formal, informal in self.idioms.items():
            pattern = r'\b' + formal + r'\b'
            text = re.sub(pattern, informal, text, flags=re.IGNORECASE)
        return text

    def create_tools(self):
        """Create StructuredTool instances"""
        return [
            StructuredTool.from_function(
                func=self.search_university,
                name="search_university",
                description="Search for university information",
            ),
            StructuredTool.from_function(
                func=self.scrape_webpage,
                name="scrape_webpage",
                description="Scrape content from a webpage",
            ),
            StructuredTool.from_function(
                func=self.generate_sop,
                name="generate_sop",
                description="Generate SOP using LLM",
            ),
            StructuredTool.from_function(
                func=self.humanize_text,
                name="humanize_text",
                description="Make text more conversational",
                args_schema =HumanizeInput  # Explicitly specify string input
            )
        ]