"""
개선된 Tree 구조 - 즉각 저장 방식에 최적화
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from classroom_segment import JudgeResponse


@dataclass
class TurnPair:
    """하나의 학생-교사 발화 쌍"""
    student_message: dict  # {'role': 'student', 'content': ...}
    teacher_message: dict  # {'role': 'teacher', 'content': ...}
    
    # Judge 결과 (나중에 채워짐)
    judge_results: Dict[str, List[JudgeResponse]] = field(default_factory=dict)  # rule_name -> List[JudgeResponse]
    pedagogical_reward: Optional[float] = None  # 이 턴의 pedagogical reward
    
    # Advantages (나중에 채워짐) - ADDED
    accuracy_advantage: Optional[float] = None  # 이 턴이 속한 노드의 accuracy advantage
    pedagogical_advantage: Optional[float] = None  # 이 턴의 pedagogical advantage (level mean과의 차이)
    combined_advantage: Optional[float] = None  # accuracy + λ * pedagogical


@dataclass
class TreeNode:
    """Tree의 각 노드 - max_group_size만큼의 대화를 담음"""
    conversation_id: str  # 이 노드가 속한 conversation의 ID (problem_idx 기반)
    node_id: int  # Tree 내에서의 고유 ID
    problem_idx: int  # 어떤 문제에 대한 노드인지
    
    # 이 노드에 속한 턴들 (max_group_size만큼)
    turn_pairs: List[TurnPair] = field(default_factory=list)
    
    # Tree 구조
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    # Rewards (최종적으로 채워짐)
    accuracy_reward: Optional[float] = None  # 0 or 1 (leaf node만)
    
    # Monte Carlo values
    v_value: Optional[float] = None  # MC estimate
    accuracy_advantage: Optional[float] = None  # V(child) - V(parent)
    
    # Pedagogical rewards (각 턴의 평균)
    avg_pedagogical_reward: Optional[float] = None  # 이 노드의 턴들의 평균
    pedagogical_advantage: Optional[float] = None  # reward - group_mean
    
    def add_turn_pair(self, student_msg: dict, teacher_msg: dict):
        """턴 쌍 추가"""
        turn = TurnPair(
            student_message=student_msg,
            teacher_message=teacher_msg
        )
        self.turn_pairs.append(turn)
    
    def get_all_messages(self) -> List[dict]:
        """이 노드의 모든 메시지를 순서대로 반환"""
        messages = []
        for turn in self.turn_pairs:
            messages.append(turn.student_message)
            messages.append(turn.teacher_message)
        return messages
    
    def compute_avg_pedagogical_reward(self):
        """이 노드의 평균 pedagogical reward 계산"""
        rewards = [
            turn.pedagogical_reward 
            for turn in self.turn_pairs 
            if turn.pedagogical_reward is not None
        ]
        if rewards:
            self.avg_pedagogical_reward = sum(rewards) / len(rewards)
        else:
            self.avg_pedagogical_reward = 0.0
    
    def is_leaf(self) -> bool:
        """Leaf node인지 확인"""
        return len(self.children) == 0


class ConversationTree:
    """
    문제별 대화 트리
    - 실시간으로 노드 추가
    - 나중에 rewards 업데이트
    - V-value 및 Advantage 계산
    """
    
    def __init__(self, problem_idx: int):
        self.problem_idx = problem_idx
        self.root: Optional[TreeNode] = None
        self.nodes: Dict[int, TreeNode] = {}  # node_id -> TreeNode
        self.next_node_id = 0  # 다음 노드 ID
        
        # Conversation ID별로 leaf nodes 추적 (reward 전파용)
        self.conversation_leaf_nodes: Dict[str, TreeNode] = {}  # conv_id -> leaf_node
        
        self.levels: List[List[TreeNode]] = []  # BFS 레벨
    
    def create_node(
        self,
        conversation_id: str,
        parent_node_id: Optional[int] = None
    ) -> TreeNode:
        """
        새로운 노드 생성 및 추가
        
        Args:
            conversation_id: 이 노드가 속한 conversation의 ID
            parent_node_id: 부모 노드의 ID (None이면 root)
        
        Returns:
            생성된 TreeNode
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = TreeNode(
            conversation_id=conversation_id,
            node_id=node_id,
            problem_idx=self.problem_idx
        )
        
        # 부모-자식 관계 설정
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
        Conversation의 현재 leaf node 업데이트
        (나중에 reward를 전파할 때 사용)
        """
        if node_id in self.nodes:
            self.conversation_leaf_nodes[conversation_id] = self.nodes[node_id]
    
    def assign_reward_to_conversation(
        self,
        conversation_id: str,
        accuracy_reward: float
    ):
        """
        Conversation의 최종 accuracy reward를 해당 leaf node에 할당
        
        Args:
            conversation_id: Conversation의 고유 ID
            accuracy_reward: 0 or 1
        """
        leaf_node = self.conversation_leaf_nodes.get(conversation_id)
        if leaf_node and leaf_node.is_leaf():
            leaf_node.accuracy_reward = accuracy_reward
    
    def build_levels(self):
        """BFS로 레벨별 노드 그룹화"""
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
        Monte Carlo 방식으로 V-value 계산
        Bottom-up: Leaf부터 시작해서 부모로 전파
        """
        for level in reversed(self.levels):
            for node in level:
                if node.is_leaf():
                    # Leaf: V = accuracy_reward
                    node.v_value = node.accuracy_reward if node.accuracy_reward is not None else 0.0
                else:
                    # Internal: V = mean(children's V)
                    child_values = [
                        child.v_value 
                        for child in node.children 
                        if child.v_value is not None
                    ]
                    node.v_value = sum(child_values) / len(child_values) if child_values else 0.0
    
    def compute_accuracy_advantages(self):
        """
        Accuracy advantage 계산: A(child) = V(child) - V(parent)
        """
        for level in self.levels:
            for node in level:
                if node.parent and node.v_value is not None and node.parent.v_value is not None:
                    node.accuracy_advantage = node.v_value - node.parent.v_value
                else:
                    node.accuracy_advantage = 0.0
    
    def compute_pedagogical_advantages(self):
        """
        Pedagogical advantage 계산 - TURN PAIR 레벨로!
        1. 각 노드의 평균 pedagogical reward 계산 (이미 계산됨)
        2. 같은 레벨의 모든 turn pair들의 pedagogical reward를 모아서 평균 계산
        3. 각 turn pair의 A = pedagogical_reward - level_mean
        """
        # 먼저 각 노드의 평균 계산
        for level in self.levels:
            for node in level:
                node.compute_avg_pedagogical_reward()
        
        # 레벨별로 모든 turn pair의 pedagogical reward 수집
        for level_idx, level in enumerate(self.levels):
            if not level:
                continue
            
            # 이 레벨의 모든 turn pair의 pedagogical reward 수집
            all_turn_rewards = []
            for node in level:
                for turn in node.turn_pairs:
                    if turn.pedagogical_reward is not None:
                        all_turn_rewards.append(turn.pedagogical_reward)
            
            if not all_turn_rewards:
                # 이 레벨에 pedagogical reward가 없으면 0으로
                for node in level:
                    for turn in node.turn_pairs:
                        turn.pedagogical_advantage = 0.0
                continue
            
            # 레벨 평균
            level_mean = sum(all_turn_rewards) / len(all_turn_rewards)
            
            # 각 turn pair의 advantage 계산
            for node in level:
                for turn in node.turn_pairs:
                    if turn.pedagogical_reward is not None:
                        turn.pedagogical_advantage = turn.pedagogical_reward - level_mean
                    else:
                        turn.pedagogical_advantage = 0.0
    
    def assign_advantages_to_turn_pairs(self, lambda_pedagogical: float = 1.0):
        """
        각 turn pair에 accuracy advantage와 combined advantage 할당
        
        Args:
            lambda_pedagogical: pedagogical advantage의 가중치
        """
        for node in self.nodes.values():
            for turn in node.turn_pairs:
                # Accuracy advantage는 노드의 값을 사용
                turn.accuracy_advantage = node.accuracy_advantage if node.accuracy_advantage is not None else 0.0
                
                # Pedagogical advantage는 이미 turn별로 계산됨
                if turn.pedagogical_advantage is None:
                    turn.pedagogical_advantage = 0.0
                
                # Combined advantage 계산
                turn.combined_advantage = (
                    turn.accuracy_advantage + 
                    lambda_pedagogical * turn.pedagogical_advantage
                )
    
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
    
    def get_turn_pairs_along_path(self, node_id: int) -> List[Dict[str, any]]:
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
                    'combined_advantage': turn.combined_advantage if turn.combined_advantage is not None else 0.0,
                })
        
        return result
    
    def get_all_turn_pair_advantages(self) -> List[Dict[str, any]]:
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
                'v_value': float
            }}
        """
        result = {}
        for node_id, node in self.nodes.items():
            result[node_id] = {
                'accuracy_advantage': node.accuracy_advantage if node.accuracy_advantage is not None else 0.0,
                'pedagogical_advantage': node.pedagogical_advantage if node.pedagogical_advantage is not None else 0.0,
                'v_value': node.v_value if node.v_value is not None else 0.0,
            }
        return result