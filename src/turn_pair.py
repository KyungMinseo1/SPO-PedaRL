from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from src.classroom_segment import JudgeResponse


@dataclass
class TurnPair:
    conversation_id: str # To identify which conversation this turn pair belongs to
    teacher_turn: int
    is_main_turn: bool  # True if this turn pair corresponds to a main teacher message, False if it's an auxiliary teacher message
    student_message: dict  # {'role': 'student', 'content': ...}
    teacher_message: dict  # {'role': 'teacher', 'content': ...}
    
    # Judge Results
    judge_results: Dict[str, List[JudgeResponse]] = field(default_factory=dict)  # rule_name -> List[JudgeResponse]
    pedagogical_reward: Optional[float] = None  # This turn's pedagogical reward
    length_reward: Optional[float] = None  # This turn's length reward
    think_reward: Optional[float] = None  # This turn's think reward
    
    # Per Conversation Advantages (Only for main turns, auxiliary turns will have None)
    accuracy_advantage: Optional[float] = None  # Conversation's accuracy advantage
    end_of_conversation_advantage: Optional[float] = None  # Conversation's end-of-conversation advantage

    # Per Turn Advantages
    pedagogical_advantage: Optional[float] = None  # This turn's pedagogical advantage
    length_advantage: Optional[float] = None  # This turn's length advantage
    
    # Depends on hyperparameter(Per Node vs Per Turn)
    think_advantage: Optional[float] = None  # This turn's or node's thinking advantage