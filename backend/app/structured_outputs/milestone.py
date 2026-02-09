from pydantic import BaseModel, Field
from typing import List


class Milestone(BaseModel):
    """Milestone schema"""

    name: str = Field(..., description="Milestone name")
    description: str = Field(..., description="Milestone description")
    tasks: List[str] = Field(..., description="List of tasks")
    isCompleted: bool = Field(..., description="Is completed")


class MilestoneOutput(BaseModel):
    """Output schema for Milestone agent"""

    milestones: List[Milestone] = Field(..., description="List of milestones")
