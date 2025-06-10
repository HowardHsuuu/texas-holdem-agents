from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
import numpy as np
import json
import os
from collections import deque, defaultdict
import time

class QLearningNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.hidden1_size = 128
        self.hidden2_size = 64
        self.hidden3_size = 32
        
        self.W1 = np.random.randn(state_size, self.hidden1_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))
        
        self.W3 = np.random.randn(self.hidden2_size, self.hidden3_size) * np.sqrt(2.0 / self.hidden2_size)
        self.b3 = np.zeros((1, self.hidden3_size))
        
        self.W4 = np.random.randn(self.hidden3_size, action_size) * np.sqrt(2.0 / self.hidden3_size)
        self.b4 = np.zeros((1, action_size))
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.m_W4, self.v_W4 = np.zeros_like(self.W4), np.zeros_like(self.W4)
        self.m_b4, self.v_b4 = np.zeros_like(self.b4), np.zeros_like(self.b4)
        
        self.t = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.leaky_relu(self.z3)
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.output = self.z4
        
        return self.output
    
    def predict(self, X):
        return self.forward(X)
    
    def adam_update(self, param, grad, m, v):
        self.t += 1
        
        m = self.beta1 * m + (1 - self.beta1) * grad
        
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_corrected = m / (1 - self.beta1 ** self.t)
        
        v_corrected = v / (1 - self.beta2 ** self.t)
        
        param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        return param, m, v
    
    def train_step(self, X, y):
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(1, -1)
                
            batch_size = X.shape[0]
            
            output = self.forward(X)
            
            loss = np.mean((output - y) ** 2)
            
            dz4 = 2 * (output - y) / batch_size
            
            dW4 = np.dot(self.a3.T, dz4)
            db4 = np.sum(dz4, axis=0, keepdims=True)
            
            da3 = np.dot(dz4, self.W4.T)
            dz3 = da3 * self.leaky_relu_derivative(self.z3)
            
            dW3 = np.dot(self.a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)
            
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * self.leaky_relu_derivative(self.z2)
            
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.leaky_relu_derivative(self.z1)
            
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            self.W4, self.m_W4, self.v_W4 = self.adam_update(self.W4, dW4, self.m_W4, self.v_W4)
            self.b4, self.m_b4, self.v_b4 = self.adam_update(self.b4, db4, self.m_b4, self.v_b4)
            
            self.W3, self.m_W3, self.v_W3 = self.adam_update(self.W3, dW3, self.m_W3, self.v_W3)
            self.b3, self.m_b3, self.v_b3 = self.adam_update(self.b3, db3, self.m_b3, self.v_b3)
            
            self.W2, self.m_W2, self.v_W2 = self.adam_update(self.W2, dW2, self.m_W2, self.v_W2)
            self.b2, self.m_b2, self.v_b2 = self.adam_update(self.b2, db2, self.m_b2, self.v_b2)
            
            self.W1, self.m_W1, self.v_W1 = self.adam_update(self.W1, dW1, self.m_W1, self.v_W1)
            self.b1, self.m_b1, self.v_b1 = self.adam_update(self.b1, db1, self.m_b1, self.v_b1)
            
            return loss
        except Exception as e:
            print(f"Training step error: {e}")
            return None
    
    def copy_weights_from(self, other_network):
        try:
            self.W1 = other_network.W1.copy()
            self.b1 = other_network.b1.copy()
            self.W2 = other_network.W2.copy()
            self.b2 = other_network.b2.copy()
            self.W3 = other_network.W3.copy()
            self.b3 = other_network.b3.copy()
            self.W4 = other_network.W4.copy()
            self.b4 = other_network.b4.copy()
        except Exception as e:
            print(f"Weight copy error: {e}")
    
    def save_weights(self, filename):
        try:
            weights = {
                'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
                'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
                'W3': self.W3.tolist(), 'b3': self.b3.tolist(),
                'W4': self.W4.tolist(), 'b4': self.b4.tolist(),
                'learning_rate': self.learning_rate,
                't': self.t,
                'architecture': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'hidden_sizes': [self.hidden1_size, self.hidden2_size, self.hidden3_size]
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(weights, f, indent=2)
            
            print(f"Saved DQN weights to {filename} (step {self.t})")
            return True
        except Exception as e:
            print(f"Failed to save DQN weights: {e}")
            return False
    
    def load_weights(self, filename):
        try:
            if not os.path.exists(filename):
                print(f"No existing DQN weights found at {filename}")
                return False
            
            with open(filename, 'r') as f:
                weights = json.load(f)
            
            arch = weights.get('architecture', {})
            if (arch.get('state_size', self.state_size) != self.state_size or
                arch.get('action_size', self.action_size) != self.action_size):
                print(f"DQN architecture mismatch, starting fresh")
                return False
            
            self.W1 = np.array(weights['W1'])
            self.b1 = np.array(weights['b1'])
            self.W2 = np.array(weights['W2'])
            self.b2 = np.array(weights['b2'])
            self.W3 = np.array(weights['W3'])
            self.b3 = np.array(weights['b3'])
            self.W4 = np.array(weights['W4'])
            self.b4 = np.array(weights['b4'])
            
            self.learning_rate = weights.get('learning_rate', 0.001)
            self.t = weights.get('t', 0)
            
            print(f"Loaded DQN weights from {filename} (step {self.t})")
            return True
        except Exception as e:
            print(f"Failed to load DQN weights: {e}")
            return False

class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        try:
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)
        except Exception as e:
            print(f"Experience push error: {e}")
    
    def sample(self, batch_size):
        try:
            if len(self.buffer) < batch_size:
                return None
            
            batch = random.sample(self.buffer, batch_size)
            
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            return states, actions, rewards, next_states, dones
        except Exception as e:
            print(f"Experience sampling error: {e}")
            return None
    
    def __len__(self):
        return len(self.buffer)

class PokerStateEncoder:
    
    @staticmethod
    def encode_state(hole_card, community_card, round_state, position, my_stack, opponent_stack, round_count):
        features = np.zeros(50)
        
        try:
            if hole_card and len(hole_card) == 2:
                features[0] = PokerStateEncoder._card_to_number(hole_card[0]) / 52.0
                features[1] = PokerStateEncoder._card_to_number(hole_card[1]) / 52.0
            
            for i, card in enumerate(community_card[:5]):
                features[2 + i] = PokerStateEncoder._card_to_number(card) / 52.0
            
            features[7] = 1.0 if position == 'SB' else 0.0
            
            total_chips = my_stack + opponent_stack
            features[8] = my_stack / total_chips if total_chips > 0 else 0.5
            features[9] = opponent_stack / total_chips if total_chips > 0 else 0.5
            features[10] = min(my_stack / 1000.0, 2.0)
            
            features[11] = round_count / 20.0
            features[12] = len(community_card) / 5.0
            features[13] = 1.0 if len(community_card) == 0 else 0.0
            
            pot_size = PokerStateEncoder._get_pot_size(round_state)
            features[14] = min(pot_size / 1000.0, 2.0)
            features[15] = pot_size / total_chips if total_chips > 0 else 0.0
            
            if hole_card and len(hole_card) == 2:
                hand_features = PokerStateEncoder._extract_hand_features_v1(hole_card, community_card)
                features[16:26] = hand_features
            
            if len(community_card) >= 3:
                board_features = PokerStateEncoder._extract_board_features(community_card)
                features[26:34] = board_features
            
            action_features = PokerStateEncoder._extract_action_features(round_state)
            features[34:42] = action_features
            
            if my_stack < 200 or opponent_stack < 200:
                features[42] = 1.0
            elif total_chips < 1600:
                features[43] = 1.0
            else:
                features[44] = 1.0
            
            features[45] = 1.0 if my_stack < opponent_stack * 0.5 else 0.0
            features[46] = 1.0 if opponent_stack < my_stack * 0.5 else 0.0
            features[47] = min(round_count / 10.0, 1.0)
            features[48] = 1.0 if pot_size > my_stack * 0.3 else 0.0
            features[49] = len(community_card) / 5.0
            
        except Exception as e:
            print(f"State encoding error: {e}")
            features = np.zeros(50)
            features[8] = 0.5
            features[10] = 0.5
        
        return features
    
    @staticmethod
    def _extract_hand_features_v1(hole_card, community_card):
        features = np.zeros(10)
        
        try:
            from game.engine.card import Card
            
            hole_cards = [Card.from_str(card) for card in hole_card]
            community_cards = [Card.from_str(card) for card in community_card]
            
            hand_score = HandEvaluator.eval_hand(hole_cards, community_cards)
            
            normalized_score = min(hand_score / 200000000.0, 1.0)
            features[0] = normalized_score
            
            hand_info = HandEvaluator.gen_hand_rank_info(hole_cards, community_cards)
            hand_strength = hand_info['hand']['strength']
            
            strength_map = {
                'STRAIGHTFLASH': 0.95, 'FOURCARD': 0.90, 'FULLHOUSE': 0.85,
                'FLASH': 0.75, 'STRAIGHT': 0.70, 'THREECARD': 0.60,
                'TWOPAIR': 0.50, 'ONEPAIR': 0.35, 'HIGHCARD': 0.20
            }
            features[1] = strength_map.get(hand_strength, 0.3)
            
            features[2] = hand_info['hand']['high'] / 14.0
            features[3] = hand_info['hand']['low'] / 14.0
            features[4] = hand_info['hole']['high'] / 14.0
            features[5] = hand_info['hole']['low'] / 14.0
            
        except Exception as e:
            features = PokerStateEncoder._extract_hand_features_basic(hole_card, community_card)
        
        return features
    
    @staticmethod
    def _extract_hand_features_basic(hole_card, community_card):
        features = np.zeros(10)
        
        try:
            all_cards = hole_card + community_card
            ranks = [PokerStateEncoder._get_rank(card) for card in all_cards]
            suits = [card[0] for card in all_cards]
            
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            
            features[0] = 1.0 if max_rank_count >= 4 else 0.0
            features[1] = 1.0 if max_rank_count >= 3 else 0.0
            features[2] = 1.0 if len([c for c in rank_counts.values() if c >= 2]) >= 2 else 0.0
            features[3] = 1.0 if max_rank_count >= 2 else 0.0
            
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            features[4] = 1.0 if max_suit_count >= 5 else 0.0
            features[5] = 1.0 if max_suit_count >= 4 else 0.0
            
            features[6] = 1.0 if PokerStateEncoder._has_straight(ranks) else 0.0
            features[7] = 1.0 if PokerStateEncoder._has_straight_draw(ranks) else 0.0
            
            hole_ranks = [PokerStateEncoder._get_rank(card) for card in hole_card] if hole_card else []
            features[8] = sum(hole_ranks) / 28.0 if hole_ranks else 0.5
            features[9] = max(hole_ranks) / 14.0 if hole_ranks else 0.5
            
        except Exception as e:
            print(f"Basic hand features error: {e}")
        
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
            
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            max_suited = max(suit_counts.values()) if suit_counts else 0
            features[2] = 1.0 if max_suited >= 3 else 0.0
            features[3] = 1.0 if max_suited >= 4 else 0.0
            
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            max_rank_count = max(rank_counts.values()) if rank_counts else 0
            features[4] = 1.0 if max_rank_count >= 2 else 0.0
            features[5] = 1.0 if max_rank_count >= 3 else 0.0
            
            high_cards = sum(1 for rank in ranks if rank >= 11)
            features[6] = high_cards / len(ranks) if len(ranks) > 0 else 0
            
            features[7] = 1.0 if PokerStateEncoder._board_has_straight_potential(ranks) else 0.0
            
        except Exception as e:
            print(f"Board features error: {e}")
        
        return features
    
    @staticmethod
    def _extract_action_features(round_state):
        features = np.zeros(8)
        
        try:
            action_histories = round_state.get('action_histories', {})
            
            total_actions = 0
            raises = 0
            calls = 0
            
            for street_actions in action_histories.values():
                if isinstance(street_actions, list):
                    for action in street_actions:
                        action_type = action.get('action', '')
                        if action_type in ['RAISE', 'CALL', 'FOLD']:
                            total_actions += 1
                            if action_type == 'RAISE':
                                raises += 1
                            elif action_type == 'CALL':
                                calls += 1
            
            if total_actions > 0:
                features[0] = raises / total_actions
                features[1] = calls / total_actions
            
            street_names = ['preflop', 'flop', 'turn', 'river']
            for i, street in enumerate(street_names):
                street_actions = action_histories.get(street, [])
                if isinstance(street_actions, list):
                    street_raises = sum(1 for a in street_actions if a.get('action') == 'RAISE')
                    features[2 + i] = min(street_raises / 3.0, 1.0)
            
            features[6] = min(total_actions / 10.0, 1.0)
            features[7] = 1.0 if total_actions > 6 else 0.0
            
        except Exception as e:
            print(f"Action features error: {e}")
        
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

class ReinforcementLearningPokerAI(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        print("Initializing Reinforcement Learning Poker AI...")
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.game_phase = 'early'
        self.games_played = 0
        
        self.epsilon = 0.15
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.996
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.state_size = 50
        self.action_size = 3
        
        self.q_network = QLearningNetwork(self.state_size, self.action_size, self.learning_rate)
        self.target_network = QLearningNetwork(self.state_size, self.action_size, self.learning_rate)
        
        self.memory = ExperienceReplay(capacity=3000)
        self.batch_size = 24
        self.target_update_frequency = 50
        self.training_frequency = 8
        
        self.steps_done = 0
        self.episode_rewards = deque(maxlen=100)
        self.training_enabled = True
        self.training_sessions = 0
        
        self.current_state = None
        self.last_action = None
        self.last_state = None
        self.episode_reward = 0
        self.hand_start_stack = 1000
        
        self.save_after_games = True
        self.save_after_hands = False
        self.backup_save_frequency = 10
        
        self._load_networks()
        
        self.opponent_stats = {
            'total_hands': 0,
            'vpip': 0.5,
            'aggression_factor': 1.0,
            'recent_actions': deque(maxlen=20),
            'showdown_hands': [],
            'fold_frequency': 0.5
        }
        
        self.performance_history = {
            'hands_played': 0,
            'hands_won': 0,
            'total_profit': 0,
            'recent_results': deque(maxlen=50),
            'training_losses': deque(maxlen=100)
        }
        
        print("RL Poker AI initialization complete!")
        self._print_status()
    
    def _print_status(self):
        print("RL Poker AI Status:")
        print(f"   Games played: {self.games_played}")
        print(f"   Training steps: {self.steps_done}")
        print(f"   Epsilon (exploration): {self.epsilon:.3f}")
        print(f"   Memory size: {len(self.memory)}")
        print(f"   Target updates: {self.steps_done // self.target_update_frequency}")
        print(f"   Save strategy: After {'games' if self.save_after_games else 'hands'}")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            
            if round_state['street'] == 'preflop' and self.last_state is None:
                self.hand_start_stack = self.my_stack
            
            current_state = PokerStateEncoder.encode_state(
                hole_card, 
                round_state.get('community_card', []),
                round_state,
                self.position,
                self.my_stack,
                self.opponent_stack,
                self.round_count
            )
            
            if self.last_state is not None and self.last_action is not None:
                reward = self._calculate_intermediate_reward(round_state)
                done = False
                
                self.memory.push(
                    self.last_state,
                    self.last_action,
                    reward,
                    current_state,
                    done
                )
                
                self.episode_reward += reward
            
            action_index = self._choose_action(current_state, valid_actions)
            action, amount = self._convert_action_index_to_poker_action(action_index, valid_actions, round_state)
            
            self.last_state = current_state.copy()
            self.last_action = action_index
            self.current_state = current_state
            
            if (self.training_enabled and 
                len(self.memory) > self.batch_size and
                self.steps_done % self.training_frequency == 0):
                loss = self._train_q_network()
                if loss is not None:
                    self.performance_history['training_losses'].append(loss)
            
            if self.steps_done % self.target_update_frequency == 0:
                self.target_network.copy_weights_from(self.q_network)
                print(f"Updated target network at step {self.steps_done}")
            
            self.steps_done += 1
            
            return action, amount
            
        except Exception as e:
            print(f"RL decision error: {e}")
            return self._safe_fallback(valid_actions)
    
    def _choose_action(self, state, valid_actions):
        
        valid_action_indices = []
        if len(valid_actions) >= 1:
            valid_action_indices.append(0)
        if len(valid_actions) >= 2:
            valid_action_indices.append(1)
        if (len(valid_actions) >= 3 and 
            isinstance(valid_actions[2]['amount'], dict) and
            valid_actions[2]['amount'].get('min', -1) != -1):
            valid_action_indices.append(2)
        
        if random.random() < self.epsilon:
            return random.choice(valid_action_indices)
        else:
            try:
                state_batch = state.reshape(1, -1)
                q_values = self.q_network.predict(state_batch)[0]
                
                masked_q_values = np.full(self.action_size, -1000.0)
                for action_idx in valid_action_indices:
                    if action_idx < len(q_values):
                        masked_q_values[action_idx] = q_values[action_idx]
                
                return np.argmax(masked_q_values)
            except Exception as e:
                print(f"Q-value prediction error: {e}")
                return random.choice(valid_action_indices)
    
    def _convert_action_index_to_poker_action(self, action_index, valid_actions, round_state):
        try:
            if action_index == 0:
                return valid_actions[0]['action'], valid_actions[0]['amount']
            elif action_index == 1:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            elif action_index == 2:
                if (len(valid_actions) > 2 and 
                    isinstance(valid_actions[2]['amount'], dict) and
                    valid_actions[2]['amount'].get('min', -1) != -1):
                    return self._size_raise_intelligently(valid_actions, round_state)
                else:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']
        except Exception as e:
            print(f"Action conversion error: {e}")
            return valid_actions[0]['action'], valid_actions[0]['amount']
    
    def _size_raise_intelligently(self, valid_actions, round_state):
        try:
            raise_info = valid_actions[2]['amount']
            min_raise = raise_info['min']
            max_raise = raise_info['max']
            
            pot_size = max(PokerStateEncoder._get_pot_size(round_state), 50)
            
            if self.game_phase == 'late':
                size_factor = 3.5
            elif self.my_stack < self.opponent_stack * 0.6:
                size_factor = 3.0
            else:
                size_factor = 2.5
            
            size_factor *= random.uniform(0.8, 1.2)
            
            target_raise = min_raise + int(pot_size * size_factor)
            final_raise = max(min_raise, min(target_raise, max_raise))
            
            return 'raise', int(final_raise)
        except Exception as e:
            print(f"Raise sizing error: {e}")
            return valid_actions[1]['action'], valid_actions[1]['amount']
    
    def _calculate_intermediate_reward(self, round_state):
        try:
            reward = 0.0
            
            if self.last_action != 0:
                reward += 0.01
            
            current_pot = PokerStateEncoder._get_pot_size(round_state)
            if current_pot > self.my_stack * 0.8:
                if self.last_action == 2:
                    reward -= 0.03
            
            if self.position == 'SB' and self.last_action == 2:
                reward += 0.02
            
            return reward
        except Exception as e:
            print(f"Intermediate reward calculation error: {e}")
            return 0.0
    
    def _calculate_final_reward(self, won, hand_result_stack):
        try:
            stack_change = hand_result_stack - self.hand_start_stack
            
            if stack_change > 0:
                reward = min(stack_change / 100.0, 5.0)
            else:
                reward = max(stack_change / 100.0, -5.0)
            
            if won:
                reward += 1.0
            
            if hand_result_stack == 0:
                reward -= 2.0
            
            return reward
        except Exception as e:
            print(f"Final reward calculation error: {e}")
            return 0.0
    
    def _train_q_network(self):
        try:
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return None
            
            states, actions, rewards, next_states, dones = batch
            
            current_q_values = self.q_network.predict(states)
            
            next_q_values = self.target_network.predict(next_states)
            
            targets = current_q_values.copy()
            
            for i in range(len(batch[0])):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            loss = self.q_network.train_step(states, targets)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.training_sessions += 1
            
            if self.training_sessions % 20 == 0:
                print(f"Training session #{self.training_sessions}, Loss: {loss:.4f}, ε: {self.epsilon:.3f}")
            
            return loss
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def _load_networks(self):
        print("Loading RL networks...")
        try:
            q_loaded = self.q_network.load_weights('rl_q_network_v1.json')
            
            target_loaded = self.target_network.load_weights('rl_target_network_v1.json')
            
            if not target_loaded and q_loaded:
                self.target_network.copy_weights_from(self.q_network)
                print("Initialized target network from main network")
            
            if os.path.exists('rl_training_state_v1.json'):
                with open('rl_training_state_v1.json', 'r') as f:
                    state = json.load(f)
                    self.epsilon = state.get('epsilon', 0.15)
                    self.steps_done = state.get('steps_done', 0)
                    self.games_played = state.get('games_played', 0)
                    self.training_sessions = state.get('training_sessions', 0)
                    
                    recent_rewards = state.get('episode_rewards', [])
                    self.episode_rewards.extend(recent_rewards[-50:])
                
                print(f"Loaded training state - Games: {self.games_played}, Steps: {self.steps_done}")
            else:
                print("No training state found, starting fresh")
                
        except Exception as e:
            print(f"Loading networks failed: {e}")
            self.target_network.copy_weights_from(self.q_network)
    
    def _save_networks(self, force_save=False):
        if not force_save and not self.save_after_games:
            return False
            
        try:
            self.q_network.save_weights('rl_q_network_v1.json')
            self.target_network.save_weights('rl_target_network_v1.json')
            
            training_state = {
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'games_played': self.games_played,
                'training_sessions': self.training_sessions,
                'episode_rewards': list(self.episode_rewards)[-50:],
                'performance': {
                    'hands_played': self.performance_history['hands_played'],
                    'hands_won': self.performance_history['hands_won'],
                    'total_profit': self.performance_history['total_profit']
                }
            }
            
            with open('rl_training_state_v1.json', 'w') as f:
                json.dump(training_state, f, indent=2)
            
            print(f"Saved RL networks and state - Games: {self.games_played}, Steps: {self.steps_done}")
            return True
        except Exception as e:
            print(f"Failed to save RL networks: {e}")
            return False
    
    def _update_game_state(self, round_state):
        try:
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
            
            if self.my_stack < 250 or self.opponent_stack < 250:
                self.game_phase = 'late'
            elif self.my_stack + self.opponent_stack < 1600:
                self.game_phase = 'middle'
            else:
                self.game_phase = 'early'
            
            self._update_opponent_stats(round_state)
        except Exception as e:
            print(f"Game state update error: {e}")
    
    def _update_opponent_stats(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            for street_actions in action_histories.values():
                if isinstance(street_actions, list):
                    for action in street_actions:
                        if action.get('uuid') != self.uuid:
                            action_type = action.get('action', '').upper()
                            if action_type:
                                self.opponent_stats['recent_actions'].append(action_type)
            
            if len(self.opponent_stats['recent_actions']) > 5:
                recent = list(self.opponent_stats['recent_actions'])[-20:]
                raises = recent.count('RAISE')
                folds = recent.count('FOLD')
                total = len([a for a in recent if a in ['RAISE', 'CALL', 'FOLD']])
                
                if total > 0:
                    self.opponent_stats['aggression_factor'] = raises / total
                    self.opponent_stats['fold_frequency'] = folds / total
        except Exception as e:
            print(f"Opponent stats update error: {e}")
    
    def _safe_fallback(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            if call_amount <= self.my_stack * 0.1:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']
        except:
            return 'fold', 0
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        print(f"Starting RL game #{self.games_played}")
        
        self.my_stack = game_info['rule']['initial_stack']
        self.opponent_stack = game_info['rule']['initial_stack']
        self.round_count = 0
        self.game_phase = 'early'
        
        self.episode_reward = 0
        self.last_state = None
        self.last_action = None
        self.hand_start_stack = self.my_stack
        
        self.opponent_stats = {
            'total_hands': 0, 'vpip': 0.5, 'aggression_factor': 1.0,
            'recent_actions': deque(maxlen=20), 'showdown_hands': [],
            'fold_frequency': 0.5
        }
        
        if self.save_after_games and self.games_played > 1:
            print(f"Auto-saving after game #{self.games_played - 1}...")
            self._save_networks()
            
            if (self.games_played - 1) % self.backup_save_frequency == 0:
                backup_name = f'rl_q_network_backup_{self.games_played - 1}.json'
                self.q_network.save_weights(backup_name)
                print(f"Backup saved: {backup_name}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.opponent_stats['total_hands'] += 1
        
        self.hand_start_stack = self.my_stack
        self.last_state = None
        self.last_action = None
        
        if round_count > 15:
            self.epsilon = max(self.epsilon_min, min(0.2, self.epsilon * 1.05))
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        try:
            if action.get('player_uuid') != self.uuid:
                action_type = action.get('action', '').upper()
                if action_type:
                    self.opponent_stats['recent_actions'].append(action_type)
        except Exception as e:
            print(f"Game update processing error: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            won = any(winner.get('uuid') == self.uuid for winner in winners)
            
            final_stack = self.my_stack
            
            if self.last_state is not None and self.last_action is not None:
                final_reward = self._calculate_final_reward(won, final_stack)
                
                dummy_next_state = np.zeros(self.state_size)
                self.memory.push(
                    self.last_state,
                    self.last_action,
                    final_reward,
                    dummy_next_state,
                    True
                )
                
                self.episode_reward += final_reward
            
            self.episode_rewards.append(self.episode_reward)
            
            self.performance_history['hands_played'] += 1
            if won:
                self.performance_history['hands_won'] += 1
            
            profit = final_stack - self.hand_start_stack
            self.performance_history['total_profit'] += profit
            self.performance_history['recent_results'].append(profit)
            
            if self.performance_history['hands_played'] % 25 == 0:
                self._print_performance_stats()
            
            self.episode_reward = 0
            self.last_state = None
            self.last_action = None
            
        except Exception as e:
            print(f"Round result processing error: {e}")
    
    def _print_performance_stats(self):
        try:
            hands_played = self.performance_history['hands_played']
            hands_won = self.performance_history['hands_won']
            total_profit = self.performance_history['total_profit']
            
            win_rate = hands_won / hands_played if hands_played > 0 else 0
            avg_profit = total_profit / hands_played if hands_played > 0 else 0
            
            recent_results = list(self.performance_history['recent_results'])[-25:]
            recent_avg = sum(recent_results) / len(recent_results) if recent_results else 0
            
            print("RL Poker AI Performance:")
            print(f"   Hands: {hands_played} | Win rate: {win_rate:.1%}")
            print(f"   Total profit: {total_profit:+.0f} | Avg per hand: {avg_profit:+.1f}")
            print(f"   Recent avg (25 hands): {recent_avg:+.1f}")
            print(f"   Exploration rate: {self.epsilon:.3f}")
            print(f"   Memory utilization: {len(self.memory)}/{self.memory.capacity}")
        except Exception as e:
            print(f"Stats printing error: {e}")
    
    def get_training_stats(self):
        try:
            recent_rewards = list(self.episode_rewards)[-50:] if self.episode_rewards else [0]
            recent_results = list(self.performance_history['recent_results'])[-25:] if self.performance_history['recent_results'] else [0]
            
            return {
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'games_played': self.games_played,
                'training_sessions': self.training_sessions,
                'avg_episode_reward': np.mean(recent_rewards),
                'avg_profit': np.mean(recent_results),
                'hands_played': self.performance_history['hands_played'],
                'win_rate': self.performance_history['hands_won'] / max(self.performance_history['hands_played'], 1),
                'memory_size': len(self.memory),
                'total_profit': self.performance_history['total_profit']
            }
        except:
            return {'error': 'Stats calculation failed'}
    
    def __del__(self):
        try:
            print("RL AI shutting down - saving final state...")
            self._save_networks(force_save=True)
            self.q_network.save_weights('rl_q_network_final.json')
            print("Final RL save completed")
        except:
            pass

def setup_ai():
    return ReinforcementLearningPokerAI()