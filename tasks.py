from crewai import Task, Agent
from typing import Dict

class SOPTasks:
    """Collection of tasks for SOP generation"""
    
    @staticmethod
    def create_research_task(agent: Agent, university: str, program: str) -> Task:
        """Create a task for researching program information"""
        return Task(
            description=f"Research {program} at {university}",
            agent=agent,
            expected_output=f"A comprehensive summary of key details about the {program} at {university}, including unique program features, research opportunities, faculty highlights, and any specific requirements or strengths of the program."
        )

    @staticmethod
    def create_writing_task(agent: Agent, student_info: Dict, university: str, program: str) -> Task:
        """Create a task for generating the SOP"""
        return Task(
            description=f"Generate a compelling Statement of Purpose for {student_info.get('name', 'the student')} applying to {program} at {university}",
            agent=agent,
            expected_output="A well-structured, compelling Statement of Purpose that is approximately 1000 words long, highlighting the student's background, achievements, motivations, and alignment with the target program. The SOP should be personal, authentic, and demonstrate a clear connection between the student's past experiences and future academic goals."
        )

    @staticmethod
    def create_humanize_task(agent: Agent) -> Task:
        """Create a task for humanizing the SOP"""
        return Task(
            description="Transform the SOP into more natural language.",
            agent=agent,
            expected_output="A revised Statement of Purpose that uses more conversational language, contractions, and idiomatic expressions while maintaining the original content, tone, and key messaging. The output should sound more natural and engaging."
        )