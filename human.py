from crewai import Crew, Agent, Task
from textwrap import dedent
from typing import Dict, Any
import pydantic
import os
import re
from langchain.tools import Tool

os.environ['OPENAI_API_KEY'] = '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'


class ContractionsSchema(pydantic.BaseModel):
    contractions: Dict[str, str]

class IdiomsSchema(pydantic.BaseModel):
    idioms: Dict[str, str]

class VoiceSchema(pydantic.BaseModel):
    target_voice: str

class TextTransformationTools:
    """Tools for text transformation that can be used by the agents"""
    
    def __init__(self):
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

    def apply_contractions(self, text: str, contractions: Dict[str, str]) -> str:
        """Apply contractions to the text"""
        for formal, contraction in contractions.items():
            pattern = r'\b' + formal + r'\b'
            text = re.sub(pattern, contraction, text, flags=re.IGNORECASE)
        return text

    def apply_idioms(self, text: str, idioms: Dict[str, str]) -> str:
        """Replace formal phrases with more casual idioms"""
        for formal, informal in idioms.items():
            pattern = r'\b' + formal + r'\b'
            text = re.sub(pattern, informal, text, flags=re.IGNORECASE)
        return text

    def transform_voice(self, text: str, target_voice: str) -> str:
        """Transform text between active and passive voice"""
        sentences = text.split('. ')
        transformed = []
        
        for sentence in sentences:
            if target_voice == 'passive':
                match = re.match(r'(\w+)\s+([\w\s]+)\s+(\w+)', sentence)
                if match:
                    transformed.append(f"{match.group(3)} was {match.group(2)} by {match.group(1)}")
                else:
                    transformed.append(sentence)
            else:
                match = re.match(r'(\w+)\s+was\s+(\w+)\s+by\s+(\w+)', sentence)
                if match:
                    transformed.append(f"{match.group(3)} {match.group(2)} {match.group(1)}")
                else:
                    transformed.append(sentence)
                    
        return '. '.join(transformed)

class TextAgents:
    """Collection of agents for text transformation"""
    
    def __init__(self):
        self.tools = TextTransformationTools()
        self._create_tools()

    def _create_tools(self):
        """Create Tool instances from the utility functions"""
        self.contractions_tool = Tool.from_function(
            func=self.tools.apply_contractions,
            name="apply_contractions",
            description="Apply contractions to make text more natural",
            args_schema=ContractionsSchema
        )
        
        self.idioms_tool = Tool.from_function(
            func=self.tools.apply_idioms,
            name="apply_idioms",
            description="Replace formal phrases with casual idioms",
            args_schema=IdiomsSchema
        )
        
        self.voice_tool = Tool.from_function(
            func=self.tools.transform_voice,
            name="transform_voice",
            description="Transform text between active and passive voice",
            args_schema=VoiceSchema
        )

    def get_humanizer_agent(self) -> Agent:
        """Create and return the humanizer agent"""
        return Agent(
            role='Text Humanizer',
            goal='Transform formal text into more natural, conversational language',
            backstory=dedent("""
                Expert in natural language processing with a focus on making text sound 
                more human and conversational. Specializes in applying contractions and 
                casual phrases while maintaining the original meaning.
            """),
            tools=[
                self.contractions_tool,
                self.idioms_tool
            ],
            verbose=True
        )

    def get_voice_agent(self) -> Agent:
        """Create and return the voice transformation agent"""
        return Agent(
            role='Voice Transformer',
            goal='Transform text between active and passive voice while preserving meaning',
            backstory=dedent("""
                Language specialist focused on sentence structure and voice transformation.
                Expert at identifying and modifying sentence voice while maintaining
                clarity and meaning.
            """),
            tools=[self.voice_tool],
            verbose=True
        )

class TextTasks:
    """Collection of tasks for text transformation"""
    
    @staticmethod
    def create_humanize_task(agent: Agent, input_text: str) -> Task:
        """Create a task for humanizing text"""
        return Task(
            description=dedent(f"""
                Transform this formal text into more natural, conversational language:
                '{input_text}'
                Apply appropriate contractions and casual phrases while maintaining the meaning.
            """),
            agent=agent,
            expected_output=""
        )

    @staticmethod
    def create_voice_task(agent: Agent, input_text: str, target_voice: str) -> Task:
        """Create a task for voice transformation"""
        return Task(
            description=dedent(f"""
                Transform the following text to {target_voice} voice:
                '{input_text}'
                Ensure the meaning is preserved while changing the voice.
            """),
            agent=agent,
            expected_output=""
        )

class TextTransformationCrew:
    """Main class to set up and run the text transformation crew"""
    
    def __init__(self):
        self.agents = TextAgents()
        
    def process_text(self, input_text: str, target_voice: str = None):
        """Process the text using the crew"""
        
        # Get agents
        humanizer_agent = self.agents.get_humanizer_agent()
        voice_agent = self.agents.get_voice_agent()
        
        # Create tasks
        tasks = []
        tasks.append(TextTasks.create_humanize_task(humanizer_agent, input_text))
        
        if target_voice:
            tasks.append(TextTasks.create_voice_task(voice_agent, input_text, target_voice))
        
        # Create and run the crew
        crew = Crew(
            agents=[humanizer_agent, voice_agent],
            tasks=tasks,
            verbose=True
        )
        
        return crew.kickoff()
def main():
    # Example usage
    crew = TextTransformationCrew()
    
    # Test text
    input_text = """
    I cannot believe that you are not coming to the meeting. 
    In addition, regarding the project timeline, we have not finished the initial phase.
    The team completed the task successfully.
    """
    
    # Process text with voice transformation
    result = crew.process_text(input_text, target_voice='active')
    print("\nOriginal text:")
    print(input_text)
    print("\nTransformed text:")
    print(result)

if __name__ == "__main__":
    main()