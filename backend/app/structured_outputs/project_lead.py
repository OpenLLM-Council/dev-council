from pydantic import BaseModel, Field
from typing import List


class SubTask(BaseModel):
    """Subtask schema"""

    task: str = Field(..., description="Task description")
    brief: str = Field(..., description="Brief description of the task")
    llm: str = Field(..., description="LLM assigned to the task")
    output: str = Field(..., description="Output of the task")
    reason: str = Field(..., description="Reason for assigning the task to the LLM")


class ProjectLeadOutput(BaseModel):
    """Output schema for Project Lead agent"""

    project_overview: str = Field(..., description="Overview of the project")
    user_requirements: str = Field(..., description="User requirements")
    subtasks: List[SubTask] = Field(..., description="List of subtasks")
    technology_stack: List[str] = Field(..., description="List of technologies used")
