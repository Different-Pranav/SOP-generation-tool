from fastapi import FastAPI, HTTPException
from typing import Dict
import uvicorn
from crewai import Crew, Process, LLM
import os

from generics.agents import StudentInfo, SOPRequest, SOPResponse
from core.agents import SOPAgents
from core.tasks import SOPTasks

llm_config = {
    "model": "groq/llama3.1-8b",
    "api_key": "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"
}


# Use environment variable for API key
os.environ['OPENAI_API_KEY'] = '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'
os.environ['GROQ_API_KEY'] = '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'


class SOPGenerationCrew:
    """Main class to coordinate SOP generation process"""

    def __init__(self, llm_config: Dict[str, str]):
        self.llm_config = llm_config
        try:
            self.agents = SOPAgents(llm_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SOP agents: {e}")

    def generate_sop(self, student_info: Dict, university: str, program: str) -> str:
        """Generate a complete, humanized SOP"""
        try:
            researcher = self.agents.get_research_agent()
            writer = self.agents.get_writer_agent()
            humanizer = self.agents.get_humanizer_agent()
            manager = self.agents.manager_agent()
            human_interaction = self.agents.interaction_agent()
        except AttributeError as e:
            raise RuntimeError(f"Error initializing agents: {e}")

        try:
            # Create tasks
            research_task = SOPTasks.create_research_task(researcher, university, program)
            writing_task = SOPTasks.create_writing_task(writer, student_info, university, program)
            humanize_task = SOPTasks.create_humanize_task(humanizer)
            interaction_task = SOPTasks.create_interaction_task(human_interaction)
        except Exception as e:
            raise ValueError(f"Failed to create SOP tasks: {e}")

        try:
            # Create crew
            crew = Crew(
                agents=[researcher, writer, humanizer,human_interaction],
                tasks=[research_task, writing_task, humanize_task,interaction_task],
                verbose=True,
                #planning=True
                manager_agent=manager,
                process=Process.hierarchical
            )

            # Execute crew
            crew_output = crew.kickoff()
        except Exception as e:
            raise RuntimeError(f"Error during SOP generation process: {e}")

        try:
            # Process and combine task outputs to generate the final SOP
            sop_parts = [task.raw for task in crew_output.tasks_output if hasattr(task, 'raw') and isinstance(task.raw, str)]
            if not sop_parts:
                raise ValueError("No valid outputs generated from tasks.")
            final_sop = "\n\n".join(sop_parts)
        except Exception as e:
            raise RuntimeError(f"Error processing SOP task outputs: {e}")

        print("SOP generation completed.")
        return final_sop




# FastAPI application
app = FastAPI(
    title="SOP Generation API",
    description="API for generating personalized Statements of Purpose",
    version="1.0.0"
)


@app.post("/generate-sop/", response_model=SOPResponse)
async def generate_sop(request: SOPRequest):
    """
    Generate a Statement of Purpose based on student information

    - **student_info**: Detailed information about the student
    - **university**: Target university
    - **program**: Target academic program
    """
    try:
        # Convert Pydantic model to dictionary
        student_info_dict = request.student_info.dict()

        # Create crew with provided or default LLM config
        crew = SOPGenerationCrew(llm_config)

        # Generate SOP
        sop = crew.generate_sop(
            student_info_dict,
            request.university,
            request.program
        )

        return {"sop": sop}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Validation error: {ve}")
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {re}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
