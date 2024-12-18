from crewai import Agent
from typing import Dict
from core.tools import SOPGenerationTools



class SOPAgents:

    """Collection of agents for SOP generation and processing"""
    

    def __init__(self, llm_config: Dict[str, str]):
        self.llm_config = llm_config
        self.tools = SOPGenerationTools(llm_config)
        self.tools_list = self.tools.create_tools()
    

    def get_research_agent(self) -> Agent:
        """Create and return the research agent"""
        return Agent(
            role='Program Researcher',
            goal='Research university programs and gather relevant information',
            backstory='An experienced academic researcher with a keen eye for detail, specializing in finding comprehensive and accurate information about university programs and academic opportunities.',
            tools=[tool for tool in self.tools_list if tool.name in ['search_university', 'scrape_webpage']],
            llm=self.tools.llm,
            verbose=True
        )


    def get_writer_agent(self) -> Agent:
        """Create and return the SOP writer agent"""
        return Agent(
            role='SOP Writer',
            goal='Generate compelling and personalized SOPs',
            backstory='A professional statement of purpose writer with extensive experience in crafting unique, persuasive narratives that highlight a student\'s academic journey and future aspirations.',
            tools=[tool for tool in self.tools_list if tool.name == 'generate_sop'],
            llm=self.tools.llm,
            verbose=True
    )


    def get_humanizer_agent(self) -> Agent:
        """Create and return the humanizer agent"""
        return Agent(
            role='Text Humanizer',
            goal='Make formal text more conversational and engaging',
            backstory='A language expert specializing in transforming formal academic writing into natural, conversational language that maintains the original message\'s integrity and passion.',
            tools=[tool for tool in self.tools_list if tool.name == 'humanize_text'],
            llm=self.tools.llm,
            verbose=True
        )
    

    # Define the manager agent
    def manager_agent(self) -> Agent:
        return Agent(
             role="Project Manager",
             goal="Efficiently manage the crew and ensure high-quality task completion",
             backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
             allow_delegation=True,
        )
    

    def interaction_agent(self) -> Agent:
        return Agent(
            role="Information Gathering Specialist",
            goal="Collect comprehensive and accurate information by "
                 "asking targeted, context-specific questions",
            backstory="An intelligent assistant skilled at identifying "
                      "information gaps and asking precise, helpful questions. "
                      "Ensures all necessary details are collected through "
                      "thoughtful and strategic interaction.",
            verbose=True,
            tools=[
                tool for tool in self.tools_list if tool.name in ['run']
            ],
            allow_delegation=False
    )