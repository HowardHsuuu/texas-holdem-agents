from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
import numpy as np
import json
import os
from collections import deque

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.training_loss = []
        self.training_count = 0
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def train_step(self, X, y):
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            
            m = X.shape[0]
            
            output = self.forward(X)
            
            loss = np.mean((output - y) ** 2)
            
            dz2 = (output - y) / m
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            
            self.training_loss.append(loss)
            self.training_count += 1
            
            return loss
            
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def predict(self, X):
        try:
            output = self.forward(X)
            return output
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([[0.5]])
    
    def save_weights(self, filepath):
        try:
            weights_data = {
                'W1': self.W1.tolist(),
                'b1': self.b1.tolist(),
                'W2': self.W2.tolist(),
                'b2': self.b2.tolist(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'training_count': self.training_count,
                'avg_loss': np.mean(self.training_loss[-10:]) if self.training_loss else 0.0
            }
            
            with open(filepath, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            print(f"Saved neural network weights to {filepath}")
            print(f"   Training steps: {self.training_count}")
            if self.training_loss:
                print(f"   Recent loss: {np.mean(self.training_loss[-10:]):.4f}")
            return True
            
        except Exception as e:
            print(f"Failed to save weights: {e}")
            return False
    
    def load_weights(self, filepath):
        try:
            if not os.path.exists(filepath):
                print(f"No existing weights found at {filepath}, starting fresh")
                return False
            
            with open(filepath, 'r') as f:
                weights_data = json.load(f)
            
            if (weights_data['input_size'] != self.input_size or
                weights_data['hidden_size'] != self.hidden_size or
                weights_data['output_size'] != self.output_size):
                print(f"Weight dimensions don't match, starting fresh")
                return False
            
            self.W1 = np.array(weights_data['W1'])
            self.b1 = np.array(weights_data['b1'])
            self.W2 = np.array(weights_data['W2'])
            self.b2 = np.array(weights_data['b2'])
            self.training_count = weights_data.get('training_count', 0)
            
            print(f"Loaded neural network weights from {filepath}")
            print(f"   Previous training steps: {self.training_count}")
            print(f"   Previous avg loss: {weights_data.get('avg_loss', 'N/A')}")
            return True
            
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return False

class NNPokerAI(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        print("Initializing Learning Poker AI...")
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.game_phase = 'early'
        self.hands_played = 0
        self.games_played = 0
        
        print("Creating neural networks...")
        self.hand_evaluator = SimpleNeuralNetwork(
            input_size=15,
            hidden_size=20,
            output_size=1,
            learning_rate=0.01
        )
        
        self.training_data = deque(maxlen=150)
        self.current_hand_data = {}
        
        self.learning_enabled = True
        self.train_every = 8
        self.min_training_data = 15
        
        self.save_strategy = 'round'
        self.save_every_rounds = 5
        self.backup_every_games = 3
        
        self.performance = {
            'hands_won': 0,
            'total_hands': 0,
            'total_chips_won': 0,
            'learning_sessions': 0,
            'games_won': 0,
            'games_total': 0
        }
        
        self.opponent_stats = {
            'aggression_factor': 1.0,
            'vpip': 0.5,
            'total_actions': 0,
            'aggressive_actions': 0,
            'fold_to_bet': 0,
            'total_bets_faced': 0
        }
        
        self._load_saved_data()
        
        print("Learning Poker AI initialized!")
        self._print_status()
    
    def _load_saved_data(self):
        print("Loading saved training data...")
        
        self.hand_evaluator.load_weights('learning_poker_weights_v1.json')
        
        try:
            if os.path.exists('learning_poker_data_v1.json'):
                with open('learning_poker_data_v1.json', 'r') as f:
                    data = json.load(f)
                    
                    saved_hands = data.get('hands', [])
                    self.training_data = deque(saved_hands[-100:], maxlen=150)
                    
                    self.performance = data.get('performance', self.performance)
                    self.opponent_stats = data.get('opponent_stats', self.opponent_stats)
                    self.hands_played = data.get('hands_played', 0)
                    self.games_played = data.get('games_played', 0)
                    
                print(f"Loaded {len(self.training_data)} training hands")
                print(f"   Games played: {self.games_played}, Hands: {self.hands_played}")
            else:
                print("No existing training data found")
        except Exception as e:
            print(f"Failed to load training data: {e}")
    
    def _save_training_data(self, force_save=False):
        if not force_save and self.save_strategy != 'round':
            return False
            
        try:
            data = {
                'hands': list(self.training_data),
                'performance': self.performance,
                'opponent_stats': self.opponent_stats,
                'hands_played': self.hands_played,
                'games_played': self.games_played,
                'save_info': {
                    'strategy': self.save_strategy,
                    'last_save_round': self.round_count,
                    'training_sessions': self.performance['learning_sessions']
                }
            }
            
            with open('learning_poker_data_v1.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.training_data)} training hands (Round {self.round_count})")
            return True
        except Exception as e:
            print(f"Failed to save training data: {e}")
            return False
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            
            if round_state['street'] == 'preflop':
                self._start_hand_data_collection(hole_card)
            
            features = self._extract_features(hole_card, round_state, valid_actions)
            
            engine_strength = self._get_engine_hand_strength(hole_card, round_state)
            nn_strength = self._get_nn_hand_strength(features)
            rule_strength = self._get_rule_hand_strength(hole_card, round_state)
            
            if engine_strength is not None:
                nn_confidence = min(self.hand_evaluator.training_count / 200.0, 0.4)
                final_strength = (0.6 * engine_strength + 
                                0.3 * nn_strength + 
                                0.1 * rule_strength)
            else:
                nn_confidence = min(self.hand_evaluator.training_count / 100.0, 0.6)
                final_strength = (nn_confidence * nn_strength + 
                                (1 - nn_confidence) * rule_strength)
            
            action, amount = self._make_decision(valid_actions, final_strength, round_state)
            
            self._record_decision(round_state['street'], features, action, amount, final_strength)
            
            return action, amount
            
        except Exception as e:
            print(f"Error in decision making: {e}")
            return self._emergency_action(valid_actions)
    
    def _get_engine_hand_strength(self, hole_card, round_state):
        try:
            from game.engine.card import Card
            
            if not hole_card or len(hole_card) != 2:
                return None
            
            hole_cards = [Card.from_str(card) for card in hole_card]
            community_str = round_state.get('community_card', [])
            community_cards = [Card.from_str(card) for card in community_str]
            
            hand_score = HandEvaluator.eval_hand(hole_cards, community_cards)
            
            normalized_strength = min(0.95, max(0.1, hand_score / 150000000.0))
            
            if normalized_strength < 0.05 or normalized_strength > 0.98:
                return None
            
            return normalized_strength
            
        except Exception as e:
            print(f"Engine evaluation failed: {e}")
            return None
    
    def _extract_features(self, hole_card, round_state, valid_actions):
        features = np.zeros(15)
        
        try:
            if hole_card and len(hole_card) == 2:
                ranks = [self._get_rank_value(card[1]) for card in hole_card]
                suits = [card[0] for card in hole_card]
                
                features[0] = max(ranks) / 14.0
                features[1] = min(ranks) / 14.0
                features[2] = 1.0 if ranks[0] == ranks[1] else 0.0
                features[3] = 1.0 if suits[0] == suits[1] else 0.0
            
            community = round_state.get('community_card', [])
            features[4] = len(community) / 5.0
            
            if community:
                comm_ranks = [self._get_rank_value(card[1]) for card in community]
                features[5] = max(comm_ranks) / 14.0 if comm_ranks else 0
                features[6] = len(set(comm_ranks)) / len(comm_ranks) if comm_ranks else 1
            
            features[7] = 1.0 if self.position == 'SB' else 0.0
            features[8] = self.round_count / 20.0
            features[9] = self.my_stack / 1000.0
            features[10] = self.opponent_stack / 1000.0
            
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            features[11] = min(pot_size / 500.0, 2.0)
            features[12] = min(call_amount / 500.0, 2.0)
            
            features[13] = min(self.opponent_stats['aggression_factor'], 3.0)
            features[14] = self.opponent_stats['vpip']
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def _get_nn_hand_strength(self, features):
        try:
            prediction = self.hand_evaluator.predict(features.reshape(1, -1))
            strength = float(prediction[0][0])
            return max(0.1, min(0.9, strength))
        except Exception as e:
            print(f"NN prediction error: {e}")
            return 0.5
    
    def _get_rule_hand_strength(self, hole_card, round_state):
        if round_state['street'] == 'preflop':
            return self._preflop_strength(hole_card)
        else:
            return self._postflop_strength(hole_card, round_state.get('community_card', []))
    
    def _preflop_strength(self, hole_card):
        if not hole_card or len(hole_card) != 2:
            return 0.3
        
        ranks = [self._get_rank_value(card[1]) for card in hole_card]
        suits = [card[0] for card in hole_card]
        
        high_rank = max(ranks)
        low_rank = min(ranks)
        is_pair = ranks[0] == ranks[1]
        is_suited = suits[0] == suits[1]
        
        if is_pair and high_rank >= 10:
            return 0.85 + (high_rank - 10) * 0.03
        elif high_rank == 14:
            return 0.6 + (low_rank / 14.0) * 0.2 + (0.05 if is_suited else 0)
        elif is_pair:
            return 0.5 + (high_rank / 14.0) * 0.2
        elif high_rank >= 10 and low_rank >= 10:
            return 0.65 + (0.05 if is_suited else 0)
        else:
            base = (high_rank + low_rank) / 28.0
            return base + (0.1 if is_suited else 0) + (0.1 if self.position == 'SB' else 0)
    
    def _postflop_strength(self, hole_card, community_cards):
        if not community_cards:
            return 0.4
        
        try:
            all_cards = hole_card + community_cards
            all_ranks = [self._get_rank_value(card[1]) for card in all_cards]
            
            rank_counts = {}
            for rank in all_ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            max_of_kind = max(rank_counts.values()) if rank_counts else 1
            
            if max_of_kind >= 4:
                return 0.95
            elif max_of_kind >= 3:
                return 0.85
            elif max_of_kind >= 2:
                our_ranks = [self._get_rank_value(card[1]) for card in hole_card]
                if any(rank_counts.get(rank, 0) >= 2 for rank in our_ranks):
                    return 0.65
                else:
                    return 0.4
            
            high_cards = sum(1 for card in all_cards if self._get_rank_value(card[1]) >= 11)
            base_strength = 0.3 + (high_cards * 0.08)
            return min(base_strength, 0.7)
            
        except Exception as e:
            print(f"Postflop evaluation error: {e}")
            return 0.4
    
    def _make_decision(self, valid_actions, hand_strength, round_state):
        try:
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            
            pot_odds = call_amount / (pot_size + call_amount) if pot_size > 0 else 0.5
            
            if call_amount >= self.my_stack * 0.7:
                return self._handle_big_decision(valid_actions, hand_strength)
            
            if hand_strength > 0.75:
                if random.random() < 0.8:
                    return self._make_raise(valid_actions, pot_size, 'value')
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            elif hand_strength > 0.55:
                if pot_odds < 0.3:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
                elif hand_strength > pot_odds + 0.15:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
                else:
                    return valid_actions[0]['action'], valid_actions[0]['amount']
            
            elif hand_strength > 0.35:
                if pot_odds < 0.25:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
                else:
                    return valid_actions[0]['action'], valid_actions[0]['amount']
            
            else:
                if (random.random() < 0.1 and pot_size < 100 and 
                    self.opponent_stats['aggression_factor'] < 0.7):
                    return self._make_raise(valid_actions, pot_size, 'bluff')
                
                return valid_actions[0]['action'], valid_actions[0]['amount']
                
        except Exception as e:
            print(f"Decision making error: {e}")
            return self._emergency_action(valid_actions)
    
    def _handle_big_decision(self, valid_actions, hand_strength):
        if hand_strength > 0.8:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        else:
            return valid_actions[0]['action'], valid_actions[0]['amount']
    
    def _make_raise(self, valid_actions, pot_size, raise_type):
        if len(valid_actions) < 3:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        
        raise_info = valid_actions[2]['amount']
        if not isinstance(raise_info, dict) or raise_info.get('min', -1) == -1:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        
        min_raise = raise_info['min']
        max_raise = raise_info['max']
        
        if raise_type == 'value':
            multiplier = 2.5
        elif raise_type == 'bluff':
            multiplier = 1.8
        else:
            multiplier = 2.0
        
        target_raise = min_raise + int(pot_size * multiplier)
        final_raise = max(min_raise, min(target_raise, max_raise))
        
        return 'raise', int(final_raise)
    
    def _start_hand_data_collection(self, hole_card):
        self.current_hand_data = {
            'hole_card': hole_card,
            'round_count': self.round_count,
            'position': self.position,
            'decisions': [],
            'final_result': None
        }
    
    def _record_decision(self, street, features, action, amount, hand_strength):
        if hasattr(self, 'current_hand_data') and self.current_hand_data:
            decision_data = {
                'street': street,
                'features': features.tolist(),
                'action': action,
                'amount': amount,
                'hand_strength': hand_strength
            }
            self.current_hand_data['decisions'].append(decision_data)
    
    def _complete_hand_data(self, won, stack_change):
        if hasattr(self, 'current_hand_data') and self.current_hand_data:
            self.current_hand_data['final_result'] = {
                'won': won,
                'chips_change': stack_change
            }
            
            self.training_data.append(dict(self.current_hand_data))
            
            if (len(self.training_data) >= self.min_training_data and 
                self.hands_played % self.train_every == 0):
                self._train_neural_network()
    
    def _train_neural_network(self):
        if not self.learning_enabled or len(self.training_data) < self.min_training_data:
            return
        
        print(f"Training neural network on {len(self.training_data)} hands...")
        
        try:
            features_list = []
            targets_list = []
            
            for hand_data in list(self.training_data)[-40:]:
                won = hand_data['final_result']['won']
                chips_change = hand_data['final_result']['chips_change']
                
                for decision in hand_data['decisions']:
                    features = np.array(decision['features'])
                    
                    if won and chips_change > 0:
                        target = 0.75
                    elif won:
                        target = 0.6
                    elif chips_change > -50:
                        target = 0.4
                    else:
                        target = 0.25
                    
                    features_list.append(features)
                    targets_list.append(target)
            
            if len(features_list) > 8:
                features_array = np.array(features_list)
                targets_array = np.array(targets_list).reshape(-1, 1)
                
                for _ in range(3):
                    indices = np.random.choice(len(features_array), 
                                             min(12, len(features_array)), 
                                             replace=False)
                    batch_features = features_array[indices]
                    batch_targets = targets_array[indices]
                    
                    loss = self.hand_evaluator.train_step(batch_features, batch_targets)
                
                self.hand_evaluator.save_weights('learning_poker_weights_v1.json')
                self._save_training_data()
                
                self.performance['learning_sessions'] += 1
                print(f"Training complete! Sessions: {self.performance['learning_sessions']}")
                
        except Exception as e:
            print(f"Training failed: {e}")
    
    def _get_rank_value(self, rank_char):
        if not rank_char:
            return 2
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank_char, 2)
    
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
            
            if self.my_stack < 300 or self.opponent_stack < 300:
                self.game_phase = 'late'
            else:
                self.game_phase = 'early'
            
            self._update_opponent_stats(round_state)
            
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _update_opponent_stats(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            for street_actions in action_histories.values():
                if isinstance(street_actions, list):
                    for action in street_actions:
                        if action.get('uuid') != self.uuid:
                            action_type = action.get('action', '').upper()
                            if action_type in ['CALL', 'RAISE', 'FOLD']:
                                self.opponent_stats['total_actions'] += 1
                                if action_type == 'RAISE':
                                    self.opponent_stats['aggressive_actions'] += 1
                                elif action_type == 'FOLD':
                                    self.opponent_stats['fold_to_bet'] += 1
                                
                                if action_type in ['CALL', 'RAISE']:
                                    self.opponent_stats['total_bets_faced'] += 1
            
            if self.opponent_stats['total_actions'] > 0:
                self.opponent_stats['aggression_factor'] = (
                    self.opponent_stats['aggressive_actions'] / 
                    self.opponent_stats['total_actions']
                )
            
            if self.opponent_stats['total_actions'] > 0:
                voluntary_actions = (self.opponent_stats['total_actions'] - 
                                   self.opponent_stats['fold_to_bet'])
                self.opponent_stats['vpip'] = voluntary_actions / self.opponent_stats['total_actions']
                
        except Exception as e:
            print(f"Error updating opponent stats: {e}")
    
    def _get_pot_size(self, round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = sum(side.get('amount', 0) for side in pot_info.get('side', []))
            return main_pot + side_pots
        except Exception as e:
            print(f"Error getting pot size: {e}")
            return 50
    
    def _emergency_action(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            if call_amount <= 10:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            return valid_actions[0]['action'], valid_actions[0]['amount']
        except:
            return 'fold', 0
    
    def _print_status(self):
        print("Learning AI Status:")
        print(f"   Save strategy: {self.save_strategy} (every {self.save_every_rounds} rounds)")
        print(f"   Training data: {len(self.training_data)} hands")
        print(f"   Network training steps: {self.hand_evaluator.training_count}")
        print(f"   Learning sessions: {self.performance['learning_sessions']}")
        print(f"   Games: {self.performance['games_won']}/{self.performance['games_total']}")
        print(f"   Win rate: {self.performance['hands_won']}/{self.performance['total_hands']}")
        if self.opponent_stats['total_actions'] > 0:
            print(f"   Opponent: Agg={self.opponent_stats['aggression_factor']:.2f}, VPIP={self.opponent_stats['vpip']:.2f}")
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        self.performance['games_total'] += 1
        
        print(f"Starting game #{self.games_played}")
        
        self.my_stack = game_info['rule']['initial_stack']
        self.opponent_stack = game_info['rule']['initial_stack']
        self.round_count = 0
        self.hands_played = 0
        
        self.opponent_stats = {
            'aggression_factor': 1.0,
            'vpip': 0.5,
            'total_actions': 0,
            'aggressive_actions': 0,
            'fold_to_bet': 0,
            'total_bets_faced': 0
        }
        
        if self.games_played % self.backup_every_games == 0:
            backup_name = f'learning_poker_backup_{self.games_played}.json'
            try:
                self.hand_evaluator.save_weights(backup_name)
                print(f"Backup saved: {backup_name}")
            except Exception as e:
                print(f"Backup save failed: {e}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.hands_played += 1
        
        try:
            for seat in seats:
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
        except Exception as e:
            print(f"Error updating stacks: {e}")
        
        if (self.save_strategy == 'round' and 
            self.round_count % self.save_every_rounds == 0 and 
            self.round_count > 0):
            print(f"Round-based save (Round {self.round_count})...")
            self.hand_evaluator.save_weights('learning_poker_weights_v1.json')
            self._save_training_data()
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        try:
            player_uuid = action.get('player_uuid')
            if player_uuid != self.uuid:
                pass
        except Exception as e:
            print(f"Error processing game update: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            won = any(winner.get('uuid') == self.uuid for winner in winners)
            
            final_stacks = {}
            for seat in round_state.get('seats', []):
                final_stacks[seat.get('uuid')] = seat.get('stack', 0)
            
            my_final_stack = final_stacks.get(self.uuid, self.my_stack)
            stack_change = my_final_stack - self.my_stack
            
            self.performance['total_hands'] += 1
            if won:
                self.performance['hands_won'] += 1
                self.performance['games_won'] += 1
            
            self.performance['total_chips_won'] += stack_change
            
            self._complete_hand_data(won, stack_change)
            
            if self.performance['total_hands'] % 15 == 0:
                self._print_status()
                
            if abs(stack_change) > 200:
                print(f"Significant result save (Stack change: {stack_change:+})...")
                self.hand_evaluator.save_weights('learning_poker_weights_v1.json')
                self._save_training_data()
                
        except Exception as e:
            print(f"Error processing round result: {e}")
    
    def __del__(self):
        try:
            print("Learning AI shutting down - saving final state...")
            self.hand_evaluator.save_weights('learning_poker_weights_v1_final.json')
            self._save_training_data(force_save=True)
            print("Final save completed")
        except:
            pass

def setup_ai():
    return NNPokerAI()