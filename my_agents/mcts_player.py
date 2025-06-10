from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
import numpy as np
import json
import math
import time
from collections import defaultdict, deque

class PolicyValueNetwork:
    
    def __init__(self, input_size=50, hidden_size=128, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        
        self.W3 = np.random.randn(hidden_size, 64) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, 64))
        
        self.W_policy = np.random.randn(64, 3) * np.sqrt(2.0 / 64)
        self.b_policy = np.zeros((1, 3))
        
        self.W_value = np.random.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b_value = np.zeros((1, 1))
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self._init_momentum()
    
    def _init_momentum(self):
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        
        self.m_W_policy, self.v_W_policy = np.zeros_like(self.W_policy), np.zeros_like(self.W_policy)
        self.m_b_policy, self.v_b_policy = np.zeros_like(self.b_policy), np.zeros_like(self.b_policy)
        self.m_W_value, self.v_W_value = np.zeros_like(self.W_value), np.zeros_like(self.W_value)
        self.m_b_value, self.v_b_value = np.zeros_like(self.b_value), np.zeros_like(self.b_value)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        
        self.z_policy = np.dot(self.a3, self.W_policy) + self.b_policy
        self.policy_output = self.softmax(self.z_policy)
        
        self.z_value = np.dot(self.a3, self.W_value) + self.b_value
        self.value_output = self.tanh(self.z_value)
        
        return self.policy_output, self.value_output
    
    def predict(self, X):
        return self.forward(X)
    
    def adam_update(self, param, grad, m, v):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_corrected = m / (1 - self.beta1 ** (self.t + 1))
        v_corrected = v / (1 - self.beta2 ** (self.t + 1))
        
        param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        return param, m, v
    
    def train_step(self, states, policy_targets, value_targets):
        batch_size = states.shape[0]
        self.t += 1
        
        policy_pred, value_pred = self.forward(states)
        
        policy_loss = -np.mean(np.sum(policy_targets * np.log(policy_pred + 1e-8), axis=1))
        value_loss = np.mean((value_pred - value_targets) ** 2)
        total_loss = policy_loss + value_loss
        
        dvalue = 2 * (value_pred - value_targets) / batch_size
        
        dW_value = np.dot(self.a3.T, dvalue)
        db_value = np.sum(dvalue, axis=0, keepdims=True)
        
        dpolicy = (policy_pred - policy_targets) / batch_size
        
        dW_policy = np.dot(self.a3.T, dpolicy)
        db_policy = np.sum(dpolicy, axis=0, keepdims=True)
        
        da3_value = np.dot(dvalue, self.W_value.T)
        da3_policy = np.dot(dpolicy, self.W_policy.T)
        da3 = da3_value + da3_policy
        
        dz3 = da3 * self.relu_derivative(self.z3)
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(states.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        self.W_value, self.m_W_value, self.v_W_value = self.adam_update(self.W_value, dW_value, self.m_W_value, self.v_W_value)
        self.b_value, self.m_b_value, self.v_b_value = self.adam_update(self.b_value, db_value, self.m_b_value, self.v_b_value)
        
        self.W_policy, self.m_W_policy, self.v_W_policy = self.adam_update(self.W_policy, dW_policy, self.m_W_policy, self.v_W_policy)
        self.b_policy, self.m_b_policy, self.v_b_policy = self.adam_update(self.b_policy, db_policy, self.m_b_policy, self.v_b_policy)
        
        self.W3, self.m_W3, self.v_W3 = self.adam_update(self.W3, dW3, self.m_W3, self.v_W3)
        self.b3, self.m_b3, self.v_b3 = self.adam_update(self.b3, db3, self.m_b3, self.v_b3)
        
        self.W2, self.m_W2, self.v_W2 = self.adam_update(self.W2, dW2, self.m_W2, self.v_W2)
        self.b2, self.m_b2, self.v_b2 = self.adam_update(self.b2, db2, self.m_b2, self.v_b2)
        
        self.W1, self.m_W1, self.v_W1 = self.adam_update(self.W1, dW1, self.m_W1, self.v_W1)
        self.b1, self.m_b1, self.v_b1 = self.adam_update(self.b1, db1, self.m_b1, self.v_b1)
        
        return policy_loss, value_loss, total_loss
    
    def save_weights(self, filename):
        try:
            weights = {
                'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
                'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
                'W3': self.W3.tolist(), 'b3': self.b3.tolist(),
                'W_policy': self.W_policy.tolist(), 'b_policy': self.b_policy.tolist(),
                'W_value': self.W_value.tolist(), 'b_value': self.b_value.tolist(),
                'learning_rate': self.learning_rate, 
                'training_steps': self.t,
                'network_architecture': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(weights, f, indent=2)
            
            print(f"Successfully saved MCTS neural network weights to {filename}")
            print(f"Training steps completed: {self.t}")
            print(f"Network size: {self.input_size} -> {self.hidden_size} -> 3+1")
            return True
            
        except Exception as e:
            print(f"Failed to save neural network weights: {e}")
            return False
    
    def load_weights(self, filename):
        try:
            import os
            if not os.path.exists(filename):
                print(f"No existing MCTS weights found at {filename}, starting with random weights")
                return False
            
            with open(filename, 'r') as f:
                weights = json.load(f)
            
            arch = weights.get('network_architecture', {})
            if (arch.get('input_size', self.input_size) != self.input_size or
                arch.get('hidden_size', self.hidden_size) != self.hidden_size):
                print(f"Network architecture mismatch in {filename}, starting fresh")
                return False
            
            self.W1 = np.array(weights['W1'])
            self.b1 = np.array(weights['b1'])
            self.W2 = np.array(weights['W2'])
            self.b2 = np.array(weights['b2'])
            self.W3 = np.array(weights['W3'])
            self.b3 = np.array(weights['b3'])
            self.W_policy = np.array(weights['W_policy'])
            self.b_policy = np.array(weights['b_policy'])
            self.W_value = np.array(weights['W_value'])
            self.b_value = np.array(weights['b_value'])
            
            self.learning_rate = weights.get('learning_rate', 0.001)
            self.t = weights.get('training_steps', 0)
            
            self._init_momentum()
            
            print(f"Successfully loaded MCTS neural network weights from {filename}")
            print(f"Previous training steps: {self.t}")
            print(f"Network architecture: {self.input_size} -> {self.hidden_size} -> 3+1")
            return True
            
        except Exception as e:
            print(f"Failed to load neural network weights: {e}")
            return False

class MCTSNode:
    
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        
        self.visits = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
        self.children = {}
        self.is_expanded = False
        
        self.valid_actions = []
    
    def is_leaf(self):
        return not self.is_expanded
    
    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                exploitation = child.value_sum / child.visits
                exploration = c_puct * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
                ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs, valid_actions):
        self.valid_actions = valid_actions
        
        for action_idx in valid_actions:
            prior_prob = action_probs[action_idx] if action_idx < len(action_probs) else 0.1
            child_state = self.state.copy()
            
            child = MCTSNode(
                state=child_state,
                parent=self,
                action=action_idx,
                prior_prob=prior_prob
            )
            
            self.children[action_idx] = child
        
        self.is_expanded = True
    
    def backup(self, value):
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature=1.0):
        if not self.children:
            return {}
        
        visits = np.array([child.visits for child in self.children.values()])
        actions = list(self.children.keys())
        
        if temperature == 0:
            best_action = actions[np.argmax(visits)]
            probs = {action: 0.0 for action in actions}
            probs[best_action] = 1.0
        else:
            if temperature == float('inf'):
                probs_array = np.ones(len(visits)) / len(visits)
            else:
                visits_temp = visits ** (1.0 / temperature)
                probs_array = visits_temp / np.sum(visits_temp)
            
            probs = {action: prob for action, prob in zip(actions, probs_array)}
        
        return probs

class MonteCarloTreeSearch:
    
    def __init__(self, neural_network, num_simulations=50, c_puct=1.0):
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, root_state, valid_actions, time_limit=1.5):
        start_time = time.time()
        
        root = MCTSNode(state=root_state)
        
        simulations_done = 0
        while simulations_done < self.num_simulations and (time.time() - start_time) < time_limit:
            node = root
            path = [node]
            
            while not node.is_leaf() and node.children:
                action, node = node.select_child(self.c_puct)
                path.append(node)
            
            if node.is_leaf():
                state_features = node.state.reshape(1, -1)
                action_probs, value = self.neural_network.predict(state_features)
                action_probs = action_probs[0]
                value = value[0][0]
                
                if valid_actions:
                    node.expand(action_probs, valid_actions)
                
                for path_node in reversed(path):
                    path_node.backup(value)
                    value = -value
            
            simulations_done += 1
        
        return root
    
    def get_action_probabilities(self, root_state, valid_actions, temperature=1.0):
        root = self.search(root_state, valid_actions)
        return root.get_action_probs(temperature)

class PokerStateEncoder:
    
    @staticmethod
    def encode_state(hole_card, community_card, round_state, position, my_stack, opponent_stack, round_count, opponent_stats=None):
        features = np.zeros(50)
        
        try:
            if hole_card and len(hole_card) == 2:
                features[0] = PokerStateEncoder._card_to_number(hole_card[0]) / 52.0
                features[1] = PokerStateEncoder._card_to_number(hole_card[1]) / 52.0
            
            for i, card in enumerate(community_card[:5]):
                features[2 + i] = PokerStateEncoder._card_to_number(card) / 52.0
            
            features[7] = 1.0 if position == 'SB' else 0.0
            features[8] = round_count / 20.0
            features[9] = len(community_card) / 5.0
            total_chips = my_stack + opponent_stack
            features[10] = my_stack / total_chips if total_chips > 0 else 0.5
            features[11] = min(my_stack / 1000.0, 2.0)
            
            pot_size = PokerStateEncoder._get_pot_size(round_state)
            features[12] = min(pot_size / 1000.0, 2.0)
            features[13] = pot_size / total_chips if total_chips > 0 else 0.0
            features[14] = 1.0 if pot_size > my_stack * 0.3 else 0.0
            
            if len(community_card) >= 3:
                hand_features = PokerStateEncoder._extract_hand_features(hole_card, community_card)
                features[15:23] = hand_features
            
            if len(community_card) >= 3:
                board_features = PokerStateEncoder._extract_board_features(community_card)
                features[23:31] = board_features
            
            if opponent_stats:
                features[31] = opponent_stats.get('vpip', 0.5)
                features[32] = opponent_stats.get('aggression_factor', 1.0)
                features[33] = min(opponent_stats.get('total_hands', 0) / 20.0, 1.0)
                features[34] = opponent_stats.get('fold_to_cbet', 0.6)
                features[35] = 1.0 if opponent_stats.get('is_tight', False) else 0.0
                features[36] = 1.0 if opponent_stats.get('is_aggressive', False) else 0.0
            
            action_features = PokerStateEncoder._extract_action_features(round_state)
            features[37:43] = action_features
            
            features[43] = 1.0 if my_stack < 200 else 0.0
            features[44] = 1.0 if opponent_stack < 200 else 0.0
            features[45] = 1.0 if round_count > 15 else 0.0
            features[46] = 1.0 if my_stack < opponent_stack * 0.5 else 0.0
            features[47] = 1.0 if opponent_stack < my_stack * 0.5 else 0.0
            features[48] = min(pot_size / max(my_stack, 1), 2.0)
            features[49] = 1.0 if len(community_card) == 5 else 0.0
            
        except Exception as e:
            print(f"Error in state encoding: {e}")
            features = np.zeros(50)
            features[10] = 0.5
        
        return features
    
    @staticmethod
    def _card_to_number(card_str):
        if len(card_str) != 2:
            return 0
        
        suit_map = {'S': 0, 'H': 13, 'D': 26, 'C': 39}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                   '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        
        suit_base = suit_map.get(card_str[0], 0)
        rank_offset = rank_map.get(card_str[1], 0)
        
        return suit_base + rank_offset
    
    @staticmethod
    def _get_pot_size(round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = sum(side.get('amount', 0) for side in pot_info.get('side', []))
            return main_pot + side_pots
        except:
            return 0
    
    @staticmethod
    def _extract_hand_features(hole_card, community_card):
        features = np.zeros(8)
        
        try:
            all_cards = hole_card + community_card
            ranks = [PokerStateEncoder._get_rank(card) for card in all_cards]
            suits = [card[0] for card in all_cards]
            
            rank_counts = defaultdict(int)
            for rank in ranks:
                rank_counts[rank] += 1
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            
            features[0] = 1.0 if max_rank_count >= 4 else 0.0
            features[1] = 1.0 if max_rank_count >= 3 else 0.0
            features[2] = 1.0 if len([c for c in rank_counts.values() if c >= 2]) >= 2 else 0.0
            features[3] = 1.0 if max_rank_count >= 2 else 0.0
            
            suit_counts = defaultdict(int)
            for suit in suits:
                suit_counts[suit] += 1
            
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            features[4] = 1.0 if max_suit_count >= 5 else 0.0
            features[5] = 1.0 if max_suit_count >= 4 else 0.0
            
            features[6] = 1.0 if PokerStateEncoder._has_straight(ranks) else 0.0
            features[7] = 1.0 if PokerStateEncoder._has_straight_draw(ranks) else 0.0
            
        except Exception as e:
            print(f"Error extracting hand features: {e}")
        
        return features
    
    @staticmethod
    def _extract_board_features(community_card):
        features = np.zeros(8)
        
        try:
            if len(community_card) < 3:
                return features
            
            ranks = [PokerStateEncoder._get_rank(card) for card in community_card]
            suits = [card[0] for card in community_card]
            
            unique_ranks = sorted(set(ranks))
            rank_spread = max(unique_ranks) - min(unique_ranks) if len(unique_ranks) > 1 else 0
            
            features[0] = min(rank_spread / 12.0, 1.0)
            features[1] = len(unique_ranks) / len(ranks)
            
            suit_counts = defaultdict(int)
            for suit in suits:
                suit_counts[suit] += 1
            
            max_suited = max(suit_counts.values()) if suit_counts else 0
            features[2] = 1.0 if max_suited >= 3 else 0.0
            features[3] = 1.0 if max_suited >= 4 else 0.0
            
            rank_counts = defaultdict(int)
            for rank in ranks:
                rank_counts[rank] += 1
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            features[4] = 1.0 if max_rank_count >= 2 else 0.0
            features[5] = 1.0 if max_rank_count >= 3 else 0.0
            
            high_cards = sum(1 for rank in ranks if rank >= 11)
            features[6] = high_cards / len(ranks)
            features[7] = 1.0 if PokerStateEncoder._board_has_straight_potential(ranks) else 0.0
            
        except Exception as e:
            print(f"Error extracting board features: {e}")
        
        return features
    
    @staticmethod
    def _extract_action_features(round_state):
        features = np.zeros(6)
        
        try:
            action_histories = round_state.get('action_histories', {})
            
            total_actions = 0
            raises = 0
            
            for street_actions in action_histories.values():
                if isinstance(street_actions, list):
                    for action in street_actions:
                        action_type = action.get('action', '')
                        if action_type in ['RAISE', 'CALL', 'FOLD']:
                            total_actions += 1
                            if action_type == 'RAISE':
                                raises += 1
            
            features[0] = raises / total_actions if total_actions > 0 else 0.0
            features[1] = min(total_actions / 8.0, 1.0)
            
            street_names = ['preflop', 'flop', 'turn', 'river']
            for i, street in enumerate(street_names):
                street_actions = action_histories.get(street, [])
                if isinstance(street_actions, list):
                    street_raises = sum(1 for a in street_actions if a.get('action') == 'RAISE')
                    features[2 + i] = min(street_raises / 2.0, 1.0)
        
        except Exception as e:
            print(f"Error extracting action features: {e}")
        
        return features
    
    @staticmethod
    def _get_rank(card_str):
        if len(card_str) < 2:
            return 2
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(card_str[1], 2)
    
    @staticmethod
    def _has_straight(ranks):
        try:
            unique_ranks = sorted(set(ranks))
            
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    return True
            
            if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
                return True
        except:
            pass
        
        return False
    
    @staticmethod
    def _has_straight_draw(ranks):
        try:
            unique_ranks = sorted(set(ranks))
            if len(unique_ranks) < 4:
                return False
            
            for i in range(len(unique_ranks) - 3):
                if unique_ranks[i+3] - unique_ranks[i] <= 4:
                    return True
        except:
            pass
        
        return False
    
    @staticmethod
    def _board_has_straight_potential(ranks):
        try:
            unique_ranks = sorted(set(ranks))
            if len(unique_ranks) < 3:
                return False
            
            consecutive = 1
            max_consecutive = 1
            
            for i in range(1, len(unique_ranks)):
                if unique_ranks[i] - unique_ranks[i-1] == 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
            
            return max_consecutive >= 3
        except:
            return False

class OpponentModeling:
    
    def __init__(self):
        self.hands_played = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.total_bets = 0
        self.total_raises = 0
        self.fold_to_bet = 0
        self.faced_bets = 0
        self.showdown_hands = 0
        self.showdown_wins = 0
        
        self.recent_actions = deque(maxlen=20)
        self.street_aggression = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
        self.street_actions = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}
    
    def update_preflop_action(self, action, amount):
        self.hands_played += 1
        
        if action in ['call', 'raise']:
            self.vpip_hands += 1
        
        if action == 'raise':
            self.pfr_hands += 1
    
    def update_action(self, action, street, amount=0):
        self.recent_actions.append((street, action, amount))
        
        if street in self.street_actions:
            self.street_actions[street] += 1
        
        if action in ['raise']:
            self.total_raises += 1
            if street in self.street_aggression:
                self.street_aggression[street] += 1
        elif action in ['call']:
            self.total_bets += 1
        elif action == 'fold':
            self.fold_to_bet += 1
        
        if action in ['call', 'raise']:
            self.faced_bets += 1
    
    def update_showdown(self, won):
        self.showdown_hands += 1
        if won:
            self.showdown_wins += 1
    
    def get_stats(self):
        stats = {}
        
        stats['vpip'] = self.vpip_hands / max(self.hands_played, 1)
        stats['pfr'] = self.pfr_hands / max(self.hands_played, 1)
        stats['total_hands'] = self.hands_played
        
        total_aggressive = self.total_raises
        total_passive = self.total_bets
        stats['aggression_factor'] = total_aggressive / max(total_passive, 1)
        
        stats['fold_to_cbet'] = self.fold_to_bet / max(self.faced_bets, 1)
        
        if self.showdown_hands > 0:
            stats['showdown_winrate'] = self.showdown_wins / self.showdown_hands
        else:
            stats['showdown_winrate'] = 0.5
        
        stats['is_tight'] = stats['vpip'] < 0.25
        stats['is_loose'] = stats['vpip'] > 0.65
        stats['is_aggressive'] = stats['aggression_factor'] > 2.0
        stats['is_passive'] = stats['aggression_factor'] < 0.5
        
        return stats

class MCTSPokerPlayer(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        
        print("Initializing MCTS Poker AI with Neural Network...")
        
        self.network = PolicyValueNetwork(input_size=50, hidden_size=128, learning_rate=0.001)
        
        self.mcts = MonteCarloTreeSearch(
            neural_network=self.network,
            num_simulations=40,
            c_puct=1.0
        )
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.round_count = 0
        self.position = None
        self.hole_card = []
        self.community_card = []
        
        self.opponent_model = OpponentModeling()
        
        self.training_data = {
            'states': deque(maxlen=200),
            'policies': deque(maxlen=200),
            'values': deque(maxlen=200),
            'rewards': deque(maxlen=200)
        }
        
        self.games_played = 0
        self.rounds_played = 0
        self.training_sessions = 0
        
        self.save_after_games = True
        self.save_after_rounds = False
        self.backup_save_frequency = 5
        
        self.conservative_games = 50
        
        self.exploration_rate = 0.12
        self.temperature_schedule = {
            'early': 1.0,
            'mid': 0.6,
            'late': 0.2
        }
        
        self._load_saved_data()
        
        print("MCTS Poker AI initialization complete!")
        self._print_status()
    
    def _load_saved_data(self):
        print("Loading saved AI data...")
        
        weights_loaded = self.network.load_weights('mcts_poker_weights_v1.json')
        
        try:
            import os
            if os.path.exists('mcts_training_data_v1.json'):
                with open('mcts_training_data_v1.json', 'r') as f:
                    data = json.load(f)
                    self.games_played = data.get('games_played', 0)
                    self.rounds_played = data.get('rounds_played', 0)
                    self.training_sessions = data.get('training_sessions', 0)
                    
                    if 'recent_training' in data:
                        recent = data['recent_training']
                        states = recent.get('states', [])
                        policies = recent.get('policies', [])
                        values = recent.get('values', [])
                        rewards = recent.get('rewards', [])
                        
                        self.training_data['states'].extend(states[-100:])
                        self.training_data['policies'].extend(policies[-100:])
                        self.training_data['values'].extend(values[-100:])
                        self.training_data['rewards'].extend(rewards[-100:])
                
                print(f"Loaded training metadata - Games: {self.games_played}, Rounds: {self.rounds_played}")
            else:
                print("No training metadata found, starting fresh")
        except Exception as e:
            print(f"Failed to load training metadata: {e}")
    
    def _save_training_data(self, force_save=False):
        if not force_save and not self.save_after_games:
            return False
            
        try:
            save_data = {
                'games_played': self.games_played,
                'rounds_played': self.rounds_played,
                'training_sessions': self.training_sessions,
                'network_info': {
                    'training_steps': self.network.t,
                    'architecture': f"{self.network.input_size}->{self.network.hidden_size}->3+1"
                },
                'performance_stats': {
                    'conservative_phase': self.games_played < self.conservative_games,
                    'training_data_size': len(self.training_data['states'])
                },
                'recent_training': {
                    'states': list(self.training_data['states'])[-50:],
                    'policies': list(self.training_data['policies'])[-50:],
                    'values': list(self.training_data['values'])[-50:],
                    'rewards': list(self.training_data['rewards'])[-50:],
                }
            }
            
            with open('mcts_training_data_v1.json', 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Saved training data - Games: {self.games_played}, Rounds: {self.rounds_played}")
            return True
            
        except Exception as e:
            print(f"Failed to save training data: {e}")
            return False
    
    def _print_status(self):
        print("MCTS Poker AI Status:")
        print(f"Games played: {self.games_played}")
        print(f"Rounds played: {self.rounds_played}")
        print(f"Training sessions: {self.training_sessions}")
        print(f"Neural network steps: {self.network.t}")
        print(f"Training data size: {len(self.training_data['states'])}")
        print(f"Save strategy: After {'games' if self.save_after_games else 'rounds'}")
        print(f"Network architecture: {self.network.input_size} -> {self.network.hidden_size} -> 3+1")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(hole_card, round_state)
            
            if self.games_played < self.conservative_games:
                return self._conservative_early_strategy(valid_actions, hole_card, round_state)
            
            state_features = self._encode_current_state(round_state)
            
            valid_action_indices = []
            for i, action_info in enumerate(valid_actions):
                action_name = action_info.get('action', '')
                if action_name in ['fold', 'call', 'raise']:
                    valid_action_indices.append(i)
            
            if not valid_action_indices or len(state_features) != 50:
                return self._fallback_strategy(valid_actions, hole_card, round_state)
            
            temperature = self._get_temperature()
            
            try:
                action_probs = self.mcts.get_action_probabilities(
                    root_state=state_features,
                    valid_actions=valid_action_indices,
                    temperature=temperature
                )
                
                if action_probs:
                    best_action_idx = max(action_probs.keys(), key=lambda k: action_probs[k])
                    action_info = valid_actions[best_action_idx]
                    action = action_info['action']
                    amount = action_info['amount']
                    
                    if action == 'raise' and isinstance(amount, dict):
                        min_raise = amount.get('min', -1)
                        max_raise = amount.get('max', -1)
                        
                        if min_raise != -1 and max_raise != -1:
                            raise_amount = self._calculate_raise_amount(min_raise, max_raise, round_state)
                            amount = raise_amount
                        else:
                            action = 'call'
                            amount = valid_actions[1]['amount']
                    
                    self._store_training_data(state_features, action_probs, valid_action_indices)
                    
                    return action, amount
                    
            except Exception as e:
                print(f"MCTS failed: {e}")
            
            return self._fallback_strategy(valid_actions, hole_card, round_state)
            
        except Exception as e:
            print(f"Critical error in declare_action: {e}")
            return self._emergency_action(valid_actions)
    
    def _conservative_early_strategy(self, valid_actions, hole_card, round_state):
        try:
            hand_strength = self._evaluate_hand_strength(hole_card, self.community_card)
            call_amount = valid_actions[1]['amount']
            
            print(f"Early learning: Hand strength {hand_strength:.2f}, Call amount {call_amount}")
            
            if hand_strength > 0.8:
                if call_amount <= 50:
                    return 'call', call_amount
                else:
                    return 'fold', 0
            elif hand_strength > 0.6 and call_amount <= 10:
                return 'call', call_amount
            else:
                return 'fold', 0
                
        except Exception as e:
            print(f"Error in conservative strategy: {e}")
            if valid_actions[1]['amount'] <= 5:
                return 'call', valid_actions[1]['amount']
            return 'fold', 0
    
    def _fallback_strategy(self, valid_actions, hole_card, round_state):
        try:
            hand_strength = self._evaluate_hand_strength_with_engine(hole_card, self.community_card)
            if hand_strength is None:
                hand_strength = self._evaluate_hand_strength(hole_card, self.community_card)
            
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 1.0
            
            if hand_strength > 0.7:
                if len(valid_actions) >= 3:
                    raise_info = valid_actions[2]['amount']
                    if isinstance(raise_info, dict) and raise_info.get('min', -1) != -1:
                        min_raise = raise_info['min']
                        max_raise = raise_info['max']
                        raise_amount = min(min_raise + (max_raise - min_raise) * 0.5, self.my_stack)
                        return 'raise', int(raise_amount)
                return 'call', call_amount
            
            elif hand_strength > 0.4:
                if pot_odds < 0.3:
                    return 'call', call_amount
                else:
                    return 'fold', 0
            
            else:
                if pot_odds < 0.15:
                    return 'call', call_amount
                else:
                    return 'fold', 0
                    
        except Exception as e:
            print(f"Error in fallback strategy: {e}")
            return 'call', valid_actions[1]['amount']
    
    def _evaluate_hand_strength_with_engine(self, hole_card, community_card):
        try:
            from game.engine.card import Card
            
            if not hole_card or len(hole_card) != 2:
                return None
            
            hole_cards = [Card.from_str(card) for card in hole_card]
            community_cards = [Card.from_str(card) for card in community_card]
            
            hand_score = HandEvaluator.eval_hand(hole_cards, community_cards)
            
            normalized_strength = min(0.95, max(0.1, hand_score / 200000000.0))
            
            if normalized_strength < 0.05 or normalized_strength > 0.98:
                return None
            
            return normalized_strength
            
        except Exception as e:
            print(f"Engine evaluation failed: {e}")
            return None
    
    def _evaluate_hand_strength(self, hole_card, community_card):
        if not hole_card or len(hole_card) != 2:
            return 0.2
        
        try:
            ranks = [self._get_rank(card) for card in hole_card]
            suits = [card[0] for card in hole_card]
            
            strength = 0.0
            
            if ranks[0] == ranks[1]:
                if ranks[0] >= 13:
                    strength += 0.5
                elif ranks[0] >= 10:
                    strength += 0.35
                elif ranks[0] >= 7:
                    strength += 0.25
                else:
                    strength += 0.15
            
            high_card_bonus = 0
            for rank in ranks:
                if rank == 14:
                    high_card_bonus += 0.15
                elif rank >= 12:
                    high_card_bonus += 0.08
                elif rank >= 10:
                    high_card_bonus += 0.04
            
            strength += high_card_bonus
            
            if suits[0] == suits[1]:
                strength += 0.05
            
            if abs(ranks[0] - ranks[1]) <= 3:
                strength += 0.03
            
            if min(ranks) >= 10 and max(ranks) >= 12:
                strength += 0.1
            
            if len(community_card) >= 3:
                strength = self._adjust_for_postflop(hole_card, community_card, strength)
            
            return min(strength, 1.0)
            
        except Exception as e:
            print(f"Error evaluating hand strength: {e}")
            return 0.3
    
    def _adjust_for_postflop(self, hole_card, community_card, preflop_strength):
        try:
            all_cards = hole_card + community_card
            all_ranks = [self._get_rank(card) for card in all_cards]
            all_suits = [card[0] for card in all_cards]
            
            rank_counts = defaultdict(int)
            for rank in all_ranks:
                rank_counts[rank] += 1
            
            max_of_kind = max(rank_counts.values()) if rank_counts else 1
            
            if max_of_kind >= 4:
                return 0.95
            elif max_of_kind >= 3:
                return 0.85
            elif len([c for c in rank_counts.values() if c >= 2]) >= 2:
                return 0.75
            elif max_of_kind >= 2:
                our_ranks = [self._get_rank(card) for card in hole_card]
                if any(rank_counts[rank] >= 2 for rank in our_ranks):
                    return 0.65
                else:
                    return 0.4
            
            suit_counts = defaultdict(int)
            for suit in all_suits:
                suit_counts[suit] += 1
            
            our_suits = [card[0] for card in hole_card]
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            
            if max_suit_count >= 5:
                for suit in our_suits:
                    if suit_counts[suit] >= 5:
                        return 0.8
                return 0.3
            
            elif max_suit_count >= 4:
                for suit in our_suits:
                    if suit_counts[suit] >= 4:
                        return preflop_strength + 0.2
            
            return max(preflop_strength * 0.7, 0.15)
            
        except Exception as e:
            print(f"Error adjusting for postflop: {e}")
            return max(preflop_strength * 0.8, 0.2)
    
    def _emergency_action(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            if call_amount <= 10:
                return 'call', call_amount
            return 'fold', 0
        except:
            return 'fold', 0
    
    def _get_rank(self, card_str):
        if len(card_str) < 2:
            return 2
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(card_str[1], 2)
    
    def _calculate_raise_amount(self, min_raise, max_raise, round_state):
        try:
            pot_size = self._get_pot_size(round_state)
            
            if pot_size < 50:
                multiplier = 0.3
            elif pot_size < 200:
                multiplier = 0.6
            else:
                multiplier = 0.8
            
            if self.my_stack < 300:
                multiplier *= 1.5
            
            raise_amount = min_raise + (max_raise - min_raise) * multiplier
            return int(max(min_raise, min(raise_amount, max_raise)))
            
        except Exception as e:
            print(f"Error calculating raise amount: {e}")
            return min_raise
    
    def _get_pot_size(self, round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = sum(side.get('amount', 0) for side in pot_info.get('side', []))
            return main_pot + side_pots
        except Exception as e:
            print(f"Error getting pot size: {e}")
            return 50
    
    def _update_game_state(self, hole_card, round_state):
        try:
            self.hole_card = hole_card if hole_card else []
            self.community_card = round_state.get('community_card', [])
            
            seats = round_state.get('seats', [])
            sb_pos = round_state.get('small_blind_pos', 0)
            
            for i, seat in enumerate(seats):
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                    if i == sb_pos:
                        self.position = 'SB'
                    else:
                        self.position = 'BB'
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
        
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _encode_current_state(self, round_state):
        try:
            opponent_stats = self.opponent_model.get_stats()
            
            return PokerStateEncoder.encode_state(
                hole_card=self.hole_card,
                community_card=self.community_card,
                round_state=round_state,
                position=self.position,
                my_stack=self.my_stack,
                opponent_stack=self.opponent_stack,
                round_count=self.round_count,
                opponent_stats=opponent_stats
            )
        except Exception as e:
            print(f"Error encoding state: {e}")
            return np.zeros(50)
    
    def _get_temperature(self):
        if self.round_count <= 5:
            return self.temperature_schedule['early']
        elif self.round_count <= 15:
            return self.temperature_schedule['mid']
        else:
            return self.temperature_schedule['late']
    
    def _store_training_data(self, state, action_probs, valid_actions):
        try:
            policy = np.zeros(3)
            for action_idx, prob in action_probs.items():
                if action_idx < 3:
                    policy[action_idx] = prob
            
            self.training_data['states'].append(state)
            self.training_data['policies'].append(policy)
            self.training_data['values'].append(0.0)
        except Exception as e:
            print(f"Error storing training data: {e}")
    
    def _train_network(self):
        if len(self.training_data['states']) < 16:
            return
        
        try:
            states = np.array(list(self.training_data['states'])[-32:])
            policies = np.array(list(self.training_data['policies'])[-32:])
            values = np.array(list(self.training_data['values'])[-32:]).reshape(-1, 1)
            
            loss = self.network.train_step(states, policies, values)
            self.training_sessions += 1
            
            if self.training_sessions % 10 == 0:
                print(f"Training session #{self.training_sessions} completed")
                
        except Exception as e:
            print(f"Training failed: {e}")
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        print(f"Starting game #{self.games_played}")
        
        self.round_count = 0
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.opponent_model = OpponentModeling()
        
        if self.save_after_games:
            if self.games_played > 1:
                print(f"Auto-saving after game #{self.games_played - 1}...")
                self.network.save_weights('mcts_poker_weights_v1.json')
                self._save_training_data()
                
                if (self.games_played - 1) % self.backup_save_frequency == 0:
                    backup_name = f'mcts_poker_backup_{self.games_played - 1}.json'
                    self.network.save_weights(backup_name)
                    print(f"Backup saved: {backup_name}")
        
        if len(self.training_data['states']) > 150:
            print("Cleaning old training data...")
            for key in self.training_data:
                recent_data = list(self.training_data[key])[-100:]
                self.training_data[key].clear()
                self.training_data[key].extend(recent_data)
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.rounds_played += 1
        self.hole_card = hole_card if hole_card else []
        self.community_card = []
        
        try:
            for seat in seats:
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
        except Exception as e:
            print(f"Error updating stacks: {e}")
        
        if self.save_after_rounds and self.rounds_played % 5 == 0:
            print(f"Round-based save after round #{self.rounds_played}...")
            self.network.save_weights('mcts_poker_weights_v1.json')
            self._save_training_data()
    
    def receive_street_start_message(self, street, round_state):
        try:
            self.community_card = round_state.get('community_card', [])
        except Exception as e:
            print(f"Error updating community cards: {e}")
    
    def receive_game_update_message(self, new_action, round_state):
        try:
            player_uuid = new_action.get('player_uuid')
            action = new_action.get('action', '').lower()
            amount = new_action.get('amount', 0)
            street = round_state.get('street', 'preflop')
            
            if player_uuid != self.uuid:
                self.opponent_model.update_action(action, street, amount)
                if street == 'preflop':
                    self.opponent_model.update_preflop_action(action, amount)
                    
        except Exception as e:
            print(f"Error processing game update: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            final_stacks = {}
            for seat in round_state.get('seats', []):
                final_stacks[seat.get('uuid')] = seat.get('stack', 0)
            
            my_final_stack = final_stacks.get(self.uuid, self.my_stack)
            stack_change = my_final_stack - self.my_stack
            reward = stack_change / 1000.0
            
            if winners:
                winner_uuid = winners[0].get('uuid')
                self.opponent_model.update_showdown(winner_uuid == self.uuid)
            
            if self.training_data['states']:
                recent_states = min(5, len(self.training_data['values']))
                for i in range(recent_states):
                    if len(self.training_data['values']) > i:
                        self.training_data['values'][-1-i] = reward
                        self.training_data['rewards'].append(reward)
            
            if (len(self.training_data['states']) >= 24 and 
                self.rounds_played % 3 == 0 and 
                self.games_played >= self.conservative_games):
                self._train_network()
            
            if self.round_count % 10 == 0:
                print(f"Round {self.round_count}: Stack {my_final_stack}, Change {stack_change:+}")
                
        except Exception as e:
            print(f"Error processing round result: {e}")
    
    def __del__(self):
        try:
            print("MCTS AI shutting down - saving final state...")
            self.network.save_weights('mcts_poker_weights_v1_final.json')
            self._save_training_data(force_save=True)
            print("Final save completed")
        except:
            pass

def setup_ai():
    return MCTSPokerPlayer()