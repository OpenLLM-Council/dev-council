"""
State definition for the coder agent.
Tracks conversation history, project context, and iteration control.
"""
from typing import List, TypedDict, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage


class CoderState(TypedDict):
    """
    State for the Coder agent within dev-council.
    Used in LangGraph for multi-turn agent interactions with checkpointing.
    
    Fields:
        messages: Full conversation history (auto-appended via operator.add).
                  All agent outputs and tool results accumulate here.
        
        project_path: Absolute path to the project directory where code will be written/modified.
                     Set by manager.py before invoking the coder agent.
        
        user_feedback: Feedback from human review during interruption nodes.
                      "stop" / "exit" / "no" → ends the graph
                      Any other string → continues with revisions
        
        iteration_count: Running count of coder→tools→coder cycles for the current task.
                        Used to guard against infinite ReAct loops (e.g., max 10 iterations).
    """
    # Full conversation history (auto-appended via operator.add)
    messages: Annotated[List[BaseMessage], operator.add]

    # Absolute path to the user's project directory
    project_path: str

    # Feedback provided by the human during interruption nodes.
    # "stop" / "exit" / "no" → ends the graph. Any other string → continues.
    user_feedback: Optional[str]

    # Running count of coder→tools→coder cycles for the current task.
    # Used to guard against infinite ReAct loops.
    iteration_count: int
