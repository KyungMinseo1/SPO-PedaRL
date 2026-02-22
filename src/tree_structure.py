"""
개선된 Tree 구조 - 즉각 저장 방식에 최적화
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from src.classroom_segment import JudgeResponse


@dataclass
class TurnPair:
    """하나의 학생-교사 발화 쌍"""
    student_message: dict  # {'role': 'student', 'content': ...}
    teacher_message: dict  # {'role': 'teacher', 'content': ...}
    
    # Judge Results
    judge_results: Dict[str, List[JudgeResponse]] = field(default_factory=dict)  # rule_name -> List[JudgeResponse]
    pedagogical_reward: Optional[float] = None  # This turn's pedagogical reward
    length_reward: Optional[float] = None  # This turn's length reward
    
    # Per Node Advantages
    accuracy_advantage: Optional[float] = None  # Node's accuracy advantage
    end_of_conversation_advantage: Optional[float] = None  # Node's end-of-conversation advantage

    # Per Turn Advantages
    pedagogical_advantage: Optional[float] = None  # This turn's pedagogical advantage (level mean과의 차이)
    length_advantage: Optional[float] = None  # This turn's length advantage (level mean과의 차이)
    
    # Depends on hyperparameter(Per Node vs Per Turn)
    think_advantage: Optional[float] = None  # This turn's or node's thinking advantage

    # Total Advantages
    combined_advantage: Optional[float] = None  # weighted sum of different advantages


@dataclass
class TreeNode:
    """Tree's nodes - Possesses max_group_size turns"""
    conversation_id: str  # Node's conversation ID # TODO: Is this necessary?
    node_id: int  # Unique node ID
    problem_idx: int  # Unique problem index
    
    # Turns this node has (size of max_group_size or less)
    turn_pairs: List[TurnPair] = field(default_factory=list)
    
    # Tree structure
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    # Monte Carlo rewards
    accuracy_reward: Optional[float] = None  # 0 or 1 (for leaf nodes)
    end_of_conversation_reward: Optional[float] = None  # 0 or 1 (for leaf nodes)
    
    # Monte Carlo values
    accuracy_v_value: Optional[float] = None  # MC estimate
    end_of_conversation_v_value: Optional[float] = None  # MC estimate
    accuracy_advantage: Optional[float] = None  # V(child) - V(parent)
    end_of_conversation_advantage: Optional[float] = None  # V(child) - V(parent)
    
    # Turn Level advantages
        # Pedagogical rewards and advantages (turn-level)
    pedagogical_advantage: Optional[float] = None  # reward - group_mean
        # Length rewards and advantages (turn-level)
    length_advantage: Optional[float] = None  # reward - group_mean
    
    def add_turn_pair(self, student_msg: dict, teacher_msg: dict):
        turn = TurnPair(
            student_message=student_msg,
            teacher_message=teacher_msg
        )
        self.turn_pairs.append(turn)
    
    def get_all_messages(self) -> List[dict]:
        """
        Returns:
            List of message dicts in order for current node.
        """
        messages = []
        for turn_pair in self.turn_pairs:
            messages.append(turn_pair.student_message)
            messages.append(turn_pair.teacher_message)
        return messages
    
    def is_leaf(self) -> bool:
        """Leaf node인지 확인"""
        return len(self.children) == 0


class ConversationTree:
    """
    ConversationTree for each problem_idx
    - Add nodes immediately as conversations are sampled (Classroom.sample_conversations)
    - Update reweards
    - Compute value & advantage functions
    """
    
    def __init__(self, problem_idx: int):
        self.problem_idx = problem_idx
        self.root: Optional[TreeNode] = None
        self.nodes: Dict[int, TreeNode] = {}  # node_id -> TreeNode
        self.next_node_id = 0  # For unique node ID generation
        
        # Finding leaf nodes for each conversation ID (for reward propagation)
        self.conversation_leaf_nodes: Dict[str, TreeNode] = {}  # conv_id -> leaf_node
        
        self.levels: List[List[TreeNode]] = []  # BFS level
    
    def create_node(
        self,
        conversation_id: str,
        parent_node_id: Optional[int] = None
    ) -> TreeNode:
        """
        Adding a new node
        
        Args:
            conversation_id: Conversation ID for this node
            parent_node_id: Parent node's ID (None = root)
        
        Returns:
            TreeNode
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = TreeNode(
            conversation_id=conversation_id, # TODO: Is this necessary?
            node_id=node_id,
            problem_idx=self.problem_idx
        )
        
        if parent_node_id is not None:
            parent_node = self.nodes.get(parent_node_id)
            if parent_node:
                node.parent = parent_node
                parent_node.children.append(node)
        else:
            # Root node
            self.root = node
        
        self.nodes[node_id] = node
        return node
    
    def update_leaf_node_mapping(self, conversation_id: str, node_id: int):
        """
        Leaf Node Update
        """
        if node_id in self.nodes:
            self.conversation_leaf_nodes[conversation_id] = self.nodes[node_id]
    
    def assign_accuracy_reward_to_conversation(
        self,
        conversation_id: str,
        accuracy_reward: float
    ):
        """
        Allocate conversation's final accuracy reward to leaf node
        
        Args:
            conversation_id: Conversation ID
            accuracy_reward: 0 or 1
        """
        leaf_node = self.conversation_leaf_nodes.get(conversation_id)
        if leaf_node and leaf_node.is_leaf():
            leaf_node.accuracy_reward = accuracy_reward
    
    def assign_end_of_conversation_reward_to_conversation(
        self,
        conversation_id: str,
        end_of_conversation_reward: float
    ):
        """
        Allocate conversation's final end-of-conversation reward to leaf node
        
        Args:
            conversation_id: Conversation ID
            end_of_conversation_reward: 0 or 1
        """
        leaf_node = self.conversation_leaf_nodes.get(conversation_id)
        if leaf_node and leaf_node.is_leaf():
            leaf_node.end_of_conversation_reward = end_of_conversation_reward
    
    def build_levels(self):
        """Group nodes with BFS Algorithm"""
        if not self.root:
            return
        
        self.levels = []
        queue = [(self.root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            
            if level >= len(self.levels):
                self.levels.append([])
            
            self.levels[level].append(node)
            
            for child in node.children:
                queue.append((child, level + 1))
    
    def compute_v_values(self):
        """
        Computing value function with Monte Carlo sampling.
        Bottom-up: propagate leaf node rewards to root.
        """
        for level in reversed(self.levels):
            for node in level:
                if node.is_leaf():
                    # Leaf: V = accuracy_reward
                    node.accuracy_v_value = node.accuracy_reward if node.accuracy_reward is not None else 0.0
                    node.end_of_conversation_v_value = node.end_of_conversation_reward if node.end_of_conversation_reward is not None else 0.0
                else:
                    # Internal: V = mean(children's V)
                    child_values = [
                        child.accuracy_v_value 
                        for child in node.children 
                        if child.accuracy_v_value is not None
                    ]
                    node.accuracy_v_value = sum(child_values) / len(child_values) if child_values else 0.0
                    
                    child_values = [
                        child.end_of_conversation_v_value 
                        for child in node.children 
                        if child.end_of_conversation_v_value is not None
                    ]
                    node.end_of_conversation_v_value = sum(child_values) / len(child_values) if child_values else 0.0
    
    def compute_accuracy_advantages(self):
        """
        Accuracy advantage computation: A(child) = V(child) - V(parent)
        """
        for level in self.levels:
            for node in level:
                if node.parent and node.accuracy_v_value is not None and node.parent.accuracy_v_value is not None:
                    node.accuracy_advantage = node.accuracy_v_value - node.parent.accuracy_v_value
                else:
                    node.accuracy_advantage = 0.0
    
    def compute_end_of_conversation_advantages(self):
        """
        End-of-conversation advantage computation: A(child) = V(child) - V(parent)
        """
        for level in self.levels:
            for node in level:
                if node.parent and node.end_of_conversation_v_value is not None and node.parent.end_of_conversation_v_value is not None:
                    node.end_of_conversation_advantage = node.end_of_conversation_v_value - node.parent.end_of_conversation_v_value
                else:
                    node.end_of_conversation_advantage = 0.0
    
    def compute_pedagogical_advantages(self):
        """
        Pedagogical advantage computation - turn-level
        - Compute level-wise average pedagogical reward
        - Each turn pair's advantage = turn_pair.pedagogical_reward - level-wise_avg
        """        
        for level_idx, level in enumerate(self.levels):
            if not level:
                continue
            
            # Collect level-wise pedagogical rewards from all turn pairs
            all_turn_rewards = []
            for node in level:
                for turn in node.turn_pairs:
                    if turn.pedagogical_reward is not None:
                        all_turn_rewards.append(turn.pedagogical_reward)
            
            if not all_turn_rewards:
                for node in level:
                    for turn in node.turn_pairs:
                        turn.pedagogical_advantage = 0.0
                continue
            
            # Level-wise average
            level_mean = sum(all_turn_rewards) / len(all_turn_rewards)
            
            # Calculate advantage for each turn pair
            for node in level:
                for turn in node.turn_pairs:
                    if turn.pedagogical_reward is not None:
                        turn.pedagogical_advantage = turn.pedagogical_reward - level_mean
                    else:
                        turn.pedagogical_advantage = 0.0

    def compute_length_advantages(self):
        """
        Length advantage computation - turn-level
        - Compute level-wise average length reward
        - Each turn pair's advantage = turn_pair.length_reward - level-wise_avg
        """        
        for level_idx, level in enumerate(self.levels):
            if not level:
                continue
            
            # Collect level-wise length rewards from all turn pairs
            all_turn_rewards = []
            for node in level:
                for turn in node.turn_pairs:
                    if turn.length_reward is not None:
                        all_turn_rewards.append(turn.length_reward)
            
            if not all_turn_rewards:
                for node in level:
                    for turn in node.turn_pairs:
                        turn.length_advantage = 0.0
                continue
            
            # Level-wise average
            level_mean = sum(all_turn_rewards) / len(all_turn_rewards)
            
            # Calculate advantage for each turn pair
            for node in level:
                for turn in node.turn_pairs:
                    if turn.length_reward is not None:
                        turn.length_advantage = turn.length_reward - level_mean
                    else:
                        turn.length_advantage = 0.0
    
    def assign_advantages_to_turn_pairs(
            self,
            # lambda_pedagogical: float = 1.0,
            reward_list: List[str] = [],
            reward_weights: List[float] = []
        ):
        """
        Allocate combined advantage to each turn pair
        """
        for node in self.nodes.values():
            for turn in node.turn_pairs:
                turn.accuracy_advantage = node.accuracy_advantage if node.accuracy_advantage is not None else 0.0
                turn.end_of_conversation_advantage = node.end_of_conversation_advantage if node.end_of_conversation_advantage is not None else 0.0

                if turn.pedagogical_advantage is None:
                    turn.pedagogical_advantage = 0.0
                if turn.length_advantage is None:
                    turn.length_advantage = 0.0
                
                # TODO: Should add thinking advantage as well (turn or node level)
                advantage_dict = {
                    "accuracy": turn.accuracy_advantage,
                    "pedagogical_alignment": turn.pedagogical_advantage,
                    "end_of_conversation": turn.end_of_conversation_advantage,
                    "length": turn.length_advantage
                }

                if reward_list and reward_weights:
                    turn.combined_advantage = sum(
                        advantage_dict[reward] * reward_weights[i]
                        for i, reward in enumerate(reward_list)
                    )
                else:
                    turn.combined_advantage = sum(advantage_dict.values())
    
    def get_path_to_node(self, node_id: int) -> List[int]:
        """
        Root부터 특정 노드까지의 경로 반환
        
        Args:
            node_id: 목표 노드 ID
            
        Returns:
            List of node IDs from root to target node
        """
        if node_id not in self.nodes:
            return []
        
        path = []
        current = self.nodes[node_id]
        
        while current is not None:
            path.append(current.node_id)
            current = current.parent
        
        # Root부터 시작하도록 reverse
        path.reverse()
        return path
    
    def get_turn_pairs_along_path(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Root부터 특정 노드까지 경로상의 모든 turn pairs 반환
        (순서대로, 중복 없이)
        
        Args:
            node_id: 목표 노드 ID
            
        Returns:
            List of turn pair info dicts along the path
        """
        path = self.get_path_to_node(node_id)
        result = []
        
        for node_id_in_path in path:
            node = self.nodes[node_id_in_path]
            for turn_idx, turn in enumerate(node.turn_pairs):
                result.append({
                    'node_id': node_id_in_path,
                    'turn_idx': turn_idx,
                    'student_message': turn.student_message,
                    'teacher_message': turn.teacher_message,
                    'pedagogical_reward': turn.pedagogical_reward if turn.pedagogical_reward is not None else 0.0,
                    'accuracy_advantage': turn.accuracy_advantage if turn.accuracy_advantage is not None else 0.0,
                    'pedagogical_advantage': turn.pedagogical_advantage if turn.pedagogical_advantage is not None else 0.0,
                    'end_of_conversation_advantage': turn.end_of_conversation_advantage if turn.end_of_conversation_advantage is not None else 0.0,
                    'length_advantage': turn.length_advantage if turn.length_advantage is not None else 0.0,
                    'combined_advantage': turn.combined_advantage if turn.combined_advantage is not None else 0.0,
                })
        
        return result
    
    def get_all_nodes_for_training(self) -> List[Dict[str, Any]]:
        """
        모든 노드를 학습용으로 반환 (Context 없음! 각 노드 독립)
        
        Returns:
            List of dicts with node info (no context)
        """
        result = []
        
        for node_id, node in self.nodes.items():
            if node_id == 0:  # virtual root skip
                continue
            
            # 현재 노드의 turn pair advantages만
            node_turn_pairs = []
            for turn_idx, turn in enumerate(node.turn_pairs):
                node_turn_pairs.append({
                    'turn_idx': turn_idx,
                    'student_message': turn.student_message,
                    'teacher_message': turn.teacher_message,
                    'pedagogical_reward': turn.pedagogical_reward if turn.pedagogical_reward is not None else 0.0,
                    'length_reward': turn.length_reward if turn.length_reward is not None else 0.0,
                    'accuracy_advantage': turn.accuracy_advantage if turn.accuracy_advantage is not None else 0.0,
                    'pedagogical_advantage': turn.pedagogical_advantage if turn.pedagogical_advantage is not None else 0.0,
                    'end_of_conversation_advantage': turn.end_of_conversation_advantage if turn.end_of_conversation_advantage is not None else 0.0,
                    'length_advantage': turn.length_advantage if turn.length_advantage is not None else 0.0,
                    'combined_advantage': turn.combined_advantage if turn.combined_advantage is not None else 0.0,
                })
            
            result.append({
                'problem_idx': self.problem_idx,
                'node_id': node_id,
                'parent_node_id': node.parent.node_id if node.parent else None,
                'node_turn_pairs': node_turn_pairs,  # Context 없음!
                'accuracy_v_value': node.accuracy_v_value if node.accuracy_v_value is not None else 0.0,
                'accuracy_advantage': node.accuracy_advantage if node.accuracy_advantage is not None else 0.0,
                'accuracy_reward': node.accuracy_reward if node.accuracy_reward is not None else 0.0,
                'end_of_conversation_v_value': node.end_of_conversation_v_value if node.end_of_conversation_v_value is not None else 0.0,
                'end_of_conversation_advantage': node.end_of_conversation_advantage if node.end_of_conversation_advantage is not None else 0.0,
                'end_of_conversation_reward': node.end_of_conversation_reward if node.end_of_conversation_reward is not None else 0.0,
            })
        
        return result
    
    def get_all_nodes_with_context(self) -> List[Dict[str, Any]]:
        """
        모든 노드를 context와 함께 반환 (학습용)
        
        Returns:
            List of dicts with node info and context messages
        """
        result = []
        
        for node_id, node in self.nodes.items():
            if node_id == 0:  # virtual root skip
                continue
            
            # 부모 경로의 메시지들 (context)
            path = self.get_path_to_node(node_id)
            context_messages = []
            
            # 부모 노드들의 메시지 (현재 노드 제외)
            for parent_node_id in path[:-1]:
                if parent_node_id == 0:  # virtual root skip
                    continue
                parent_node = self.nodes[parent_node_id]
                context_messages.extend(parent_node.get_all_messages())
            
            # 현재 노드의 turn pair advantages
            node_turn_pairs = []
            for turn_idx, turn in enumerate(node.turn_pairs):
                node_turn_pairs.append({
                    'turn_idx': turn_idx,
                    'student_message': turn.student_message,
                    'teacher_message': turn.teacher_message,
                    'pedagogical_reward': turn.pedagogical_reward if turn.pedagogical_reward is not None else 0.0,
                    'accuracy_advantage': turn.accuracy_advantage if turn.accuracy_advantage is not None else 0.0,
                    'pedagogical_advantage': turn.pedagogical_advantage if turn.pedagogical_advantage is not None else 0.0,
                    'combined_advantage': turn.combined_advantage if turn.combined_advantage is not None else 0.0,
                })
            
            result.append({
                'problem_idx': self.problem_idx,
                'node_id': node_id,
                'parent_node_id': node.parent.node_id if node.parent else None,
                'context_messages': context_messages,
                'node_turn_pairs': node_turn_pairs,
                'accuracy_v_value': node.accuracy_v_value if node.accuracy_v_value is not None else 0.0,
                'accuracy_advantage': node.accuracy_advantage if node.accuracy_advantage is not None else 0.0,
                'accuracy_reward': node.accuracy_reward if node.accuracy_reward is not None else 0.0,
            })
        
        return result
    
    def count_total_nodes(self) -> int:
        """Virtual root 제외한 총 노드 수 반환"""
        return len(self.nodes) - 1  # -1 for virtual root
    
    def get_all_turn_pair_advantages(self) -> List[Dict[str, Any]]:
        """
        모든 turn pair의 advantage 정보 반환
        
        Returns:
            List of dicts with turn pair info and advantages
        """
        result = []
        for node in self.nodes.values():
            for turn_idx, turn in enumerate(node.turn_pairs):
                result.append({
                    'node_id': node.node_id,
                    'turn_idx': turn_idx,
                    'student_message': turn.student_message,
                    'teacher_message': turn.teacher_message,
                    'pedagogical_reward': turn.pedagogical_reward if turn.pedagogical_reward is not None else 0.0,
                    'accuracy_advantage': turn.accuracy_advantage if turn.accuracy_advantage is not None else 0.0,
                    'pedagogical_advantage': turn.pedagogical_advantage if turn.pedagogical_advantage is not None else 0.0,
                    'length_advantage': turn.length_advantage if turn.length_advantage is not None else 0.0,
                    'end_of_conversation_advantage': turn.end_of_conversation_advantage if turn.end_of_conversation_advantage is not None else 0.0,
                    'combined_advantage': turn.combined_advantage if turn.combined_advantage is not None else 0.0,
                })
        return result
    
    def get_all_advantages(self) -> Dict[int, Dict[str, float]]:
        """
        모든 노드의 advantage 반환
        
        Returns:
            {node_id: {
                'accuracy_advantage': float,
                'pedagogical_advantage': float,
                'combined_advantage': float,
                'v_value': float
            }}
        """
        result = {}
        for node_id, node in self.nodes.items():
            result[node_id] = {
                'accuracy_advantage': node.accuracy_advantage if node.accuracy_advantage is not None else 0.0,
                'pedagogical_advantage': node.pedagogical_advantage if node.pedagogical_advantage is not None else 0.0,
                'end_of_conversation_advantage': node.end_of_conversation_advantage if node.end_of_conversation_advantage is not None else 0.0,
                'length_advantage': node.length_advantage if node.length_advantage is not None else 0.0,
                'accuracy_v_value': node.accuracy_v_value if node.accuracy_v_value is not None else 0.0,
                'end_of_conversation_v_value': node.end_of_conversation_v_value if node.end_of_conversation_v_value is not None else 0.0,
            }
        return result