import re
import os
from openai import OpenAI
import logging
import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import StructuredTool
from crewai import LLM
from litellm import completion
from pydantic import BaseModel, Field, ValidationError

RUNPOD_API_KEY = 'rpa_GK378GQ1KXFODHN6OF76VZ334RGI03N3EWAUUXNB1efd24'

BASE_URL = "https://api.runpod.ai/v2/pkrvrt52fitz9x/openai/v1"

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SOPGenerationTools:
    """Enhanced tools for SOP generation, web scraping, and text transformation"""

    def __init__(self, config: Dict[str, str]):
        """
        Initialize SOPGenerationTools with configuration
        
        :param config: Dictionary containing configuration parameters
        """
        try:
            # Validate configuration
            self.validate_config(config)
            
            # Initialize search and LLM
            self.search = DuckDuckGoSearchRun()
            self.llm = OpenAI(
                        base_url=BASE_URL,
                        api_key=RUNPOD_API_KEY,
                    )
            
            # Contractions and idioms
            self.contractions = {
                'cannot': "can't", 'will not': "won't", 'shall not': "shan't",
                'do not': "don't", 'does not': "doesn't", 'did not': "didn't",
                'is not': "isn't", 'are not': "aren't", 'was not': "wasn't",
                'have not': "haven't", 'has not': "hasn't", 
                'I am': "I'm", 'you are': "you're", 'they are': "they're"
            }
            
            self.idioms = {
                'in addition': "on top of that", 'for example': "like",
                'therefore': "so", 'however': "though", 
                'nevertheless': "still", 'subsequently': "then",
                'furthermore': "also", 'in conclusion': "to wrap things up"
            }
            
            logger.info("SOPGenerationTools initialized successfully")
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def validate_config(self, config: Dict[str, str]):
        """
        Validate configuration parameters
        
        :param config: Configuration dictionary
        :raises ValidationError: If configuration is invalid
        """
        required_keys = ['model', 'api_key']
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValidationError(f"Missing required configuration: {key}")

    def safe_input_processing(self, input_data: Any) -> str:
        """
        Safely process input to extract string value
        
        :param input_data: Input data of any type
        :return: Processed string input
        """
        if isinstance(input_data, dict):
            # Extract string from dictionary
            return (input_data.get('query') or 
                    input_data.get('description') or 
                    input_data.get('url') or 
                    str(input_data))
        return str(input_data)

    def search_university(self, query: Any) -> str:
        """
        Search for university information
        
        :param query: Search query
        :return: Search results
        """
        try:
            processed_query = self.safe_input_processing(query)
            
            if not processed_query:
                raise ValueError("Empty search query")
            
            results = self.search.run(processed_query)
            logger.info(f"University search completed: {processed_query}")
            return results
        
        except Exception as e:
            logger.error(f"University search error: {e}")
            return f"Search error: {str(e)}"

    def scrape_webpage(self, url: Any) -> str:
        """
        Scrape content from a webpage
        
        :param url: Webpage URL
        :return: Scraped text content
        """
        try:
            processed_url = self.safe_input_processing(url)
            
            # Validate URL
            if not processed_url.startswith(('http://', 'https://')):
                raise ValueError("Invalid URL format")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(processed_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            scraped_content = ' '.join(chunk for chunk in chunks if chunk)
            logger.info(f"Webpage scraped successfully: {processed_url}")
            return scraped_content
        
        except Exception as e:
            logger.error(f"Webpage scraping error: {e}")
            return f"Scraping error: {str(e)}"

    def generate_sop(self, student_info: Dict, program_info: Dict, research_info: Dict = None) -> str:
        """
        Generate Statement of Purpose with comprehensive guidelines
        
        :param student_info: Student background information
        :param program_info: Target program details
        :param research_info: Optional research information about the program/university
        :return: Generated SOP
        """
        try:
            # Validate inputs
            if not student_info or not program_info:
                raise ValueError("Incomplete input data")
            
            # Prepare research information if available
            research_details = f"Research Insights:\n{research_info}" if research_info else ""
            
            # Comprehensive SOP generation prompt with strict guidelines
            prompt = f"""Generate a Statement of Purpose following these strict guidelines:
    
    IMPORTANT INSTRUCTIONS:
    - Maximum two pages
    - Strictly five paragraphs
    - Use original content (No plagiarism)
    - Adhere to specific word limits for each paragraph
    
    PARAGRAPH 1: INTRODUCTION (100 words max)
    - Personal background
    - Family context
    - Academic qualifications
    - Work experience
    - Reason for choosing the field
    
    Key Questions to Address:
    - Who are you?
    - Where are you from?
    - What are your academic achievements?
    - What is your current professional designation?
    - Why this specific field of study?
    
    PARAGRAPH 2: CHOSEN COURSE (300 words max)
    - Name of the chosen course
    - Motivation for selecting the program
    - Relevance to academic/professional background
    - Program structure understanding
    - Career alignment
    
    Key Questions to Address:
    - What motivated you to choose this field?
    - How does the course relate to your background?
    - If changing career path, explain why
    - What do you know about the program?
    - What aspects of the program structure appeal to you?
    
    PARAGRAPH 3: UNIVERSITY SELECTION (300 words max)
    - Research about the university
    - Why this specific university
    - Unique attributes compared to other institutions
    - Program-specific insights
    
    Key Questions to Address:
    - What research led you to this university?
    - Why this university over others in the country?
    - What specific knowledge do you have about the institution?
    - How does the university align with your academic goals?
    
    PARAGRAPH 4: COUNTRY SELECTION (300 words max)
    - Reasons for choosing the country
    - Educational system comparison
    - Benefits of studying in this country
    - Motivation beyond Indian education
    
    Key Questions to Address:
    - Why this specific country?
    - How is the education system different?
    - What unique benefits does this country offer?
    - Why not pursue the same program in India?
    - How will this country's educational approach benefit you?
    
    PARAGRAPH 5: FUTURE PLANS (100 words max)
    - Career outcomes
    - Knowledge and skills acquisition
    - Post-degree professional objectives
    - Long-term career vision
    
    Key Questions to Address:
    - What skills will you acquire?
    - What are your career expectations?
    - What kind of job do you anticipate?
    - How does this program support your long-term goals?
    
    ADDITIONAL CONTEXT:
    Student Background:
    {student_info}
    
    Target Program:
    {program_info}
    
    {research_details}
    
    CRITICAL GUIDELINES:
    1. Write in first person
    2. Maintain a personal, authentic voice
    3. Be specific and genuine
    4. Proofread for originality
    5. Avoid generic statements
    
    Generate a compelling, original Statement of Purpose that showcases your unique journey, motivations, and aspirations."""
            
            client =  OpenAI(
                        base_url=BASE_URL,
                        api_key=RUNPOD_API_KEY,
                    )
            
            
            response = completion.create(
                model="openai/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            
            generated_sop = response.choices[0].message.content
            logger.info("SOP generated successfully with comprehensive guidelines")
            return generated_sop
        
        except Exception as e:
            logger.error(f"SOP generation error: {e}")
            return f"SOP generation error: {str(e)}"
        
    def humanize_text(self, text: str) -> str:
        """
        Make text more conversational
        
        :param text: Input text
        :return: Conversationalized text
        """
        try:
            # Apply contractions
            for formal, contraction in self.contractions.items():
                text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
            
            # Apply idioms
            for formal, informal in self.idioms.items():
                text = re.sub(r'\b' + formal + r'\b', informal, text, flags=re.IGNORECASE)
            
            logger.info("Text humanization completed")
            return text
        
        except Exception as e:
            logger.error(f"Text humanization error: {e}")
            return text

    def create_tools(self) -> List[StructuredTool]:
        """
        Create structured tools
        
        :return: List of StructuredTools
        """
        return [
            StructuredTool.from_function(
                func=self.search_university,
                name="search_university",
                description="Search for university information"
            ),
            StructuredTool.from_function(
                func=self.scrape_webpage,
                name="scrape_webpage",
                description="Scrape content from a webpage"
            ),
            StructuredTool.from_function(
                func=self.generate_sop,
                name="generate_sop",
                description="Generate SOP using LLM"
            ),
            StructuredTool.from_function(
                func=self.humanize_text,
                name="humanize_text",
                description="Make text more conversational"
            )
        ]