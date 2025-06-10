from game.players import BasePokerPlayer
import random
import math

class RulePokerAI(BasePokerPlayer):
    def __init__(self):
        self.opponent_model = {
            'vpip': 0.5,
            'pfr': 0.3,
            'aggression_factor': 1.0,
            'fold_to_cbet': 0.6,
            'total_hands': 0,
            'baseline_type': 'unknown',
            'tightness_level': 0.5,
            'recent_actions': [],
            'showdown_hands': []
        }
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.game_phase = 'early'
        
        self.aggression_level = 1.2
        self.bluff_frequency = 0.2
        self.value_bet_sizing = 0.6
        self.bluff_bet_sizing = 0.45
        
        self.short_stack_threshold = 300
        self.push_fold_threshold = 150
        
        self.baseline_adjustments = {
            'tight': {'fold_threshold': -0.1, 'raise_frequency': +0.2, 'bluff_freq': +0.15},
            'loose': {'fold_threshold': +0.05, 'raise_frequency': -0.1, 'bluff_freq': -0.1},
            'aggressive': {'fold_threshold': +0.1, 'raise_frequency': -0.05, 'value_size': +0.2},
            'passive': {'fold_threshold': -0.05, 'raise_frequency': +0.1, 'bluff_freq': +0.1}
        }
        
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            self._detect_baseline_type()
            
            hand_strength = self._evaluate_hand_strength(hole_card, round_state)
            
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            pot_odds = call_amount / (pot_size + call_amount) if pot_size > 0 else 0
            
            if self.my_stack <= self.push_fold_threshold:
                return self._push_fold_strategy(valid_actions, hole_card, hand_strength)
            elif self.my_stack <= self.short_stack_threshold:
                return self._short_stack_strategy(valid_actions, hole_card, round_state, hand_strength)
            
            street = round_state['street']
            if street == 'preflop':
                return self._improved_preflop_decision(valid_actions, hole_card, round_state, hand_strength)
            else:
                return self._improved_postflop_decision(valid_actions, hole_card, round_state, hand_strength, pot_odds)
                
        except Exception as e:
            print(f"Error in declare_action: {e}")
            return self._safe_action(valid_actions)
    
    def _detect_baseline_type(self):
        if len(self.opponent_model['recent_actions']) < 5:
            return
        
        recent = self.opponent_model['recent_actions'][-10:]
        
        aggressive_actions = sum(1 for a in recent if a.get('action') == 'RAISE')
        passive_actions = sum(1 for a in recent if a.get('action') == 'CALL')
        folds = sum(1 for a in recent if a.get('action') == 'FOLD')
        
        total_actions = len(recent)
        if total_actions == 0:
            return
        
        aggression_rate = aggressive_actions / total_actions
        fold_rate = folds / total_actions
        
        if fold_rate > 0.6:
            self.opponent_model['baseline_type'] = 'tight'
            self.opponent_model['tightness_level'] = 0.8
        elif fold_rate < 0.3:
            self.opponent_model['baseline_type'] = 'loose'
            self.opponent_model['tightness_level'] = 0.2
        elif aggression_rate > 0.4:
            self.opponent_model['baseline_type'] = 'aggressive'
        elif aggression_rate < 0.15:
            self.opponent_model['baseline_type'] = 'passive'
        else:
            self.opponent_model['baseline_type'] = 'balanced'
    
    def _push_fold_strategy(self, valid_actions, hole_card, hand_strength):
        push_threshold = 0.35
        
        if self.position == 'SB':
            push_threshold = 0.25
        
        baseline_type = self.opponent_model['baseline_type']
        if baseline_type == 'tight':
            push_threshold -= 0.1
        elif baseline_type == 'loose':
            push_threshold += 0.05
        
        if hand_strength >= push_threshold:
            if len(valid_actions) >= 3 and valid_actions[2]['amount']['min'] != -1:
                max_raise = valid_actions[2]['amount']['max']
                return 'raise', max_raise
            else:
                return 'call', valid_actions[1]['amount']
        else:
            return 'fold', 0
    
    def _short_stack_strategy(self, valid_actions, hole_card, round_state, hand_strength):
        call_amount = valid_actions[1]['amount']
        
        fold_threshold = 0.2
        push_threshold = 0.4
        
        baseline_type = self.opponent_model['baseline_type']
        if baseline_type == 'tight':
            fold_threshold -= 0.05
            push_threshold -= 0.1
        elif baseline_type == 'aggressive':
            fold_threshold += 0.05
            push_threshold += 0.05
        
        if hand_strength < fold_threshold:
            return 'fold', 0
        elif hand_strength > push_threshold:
            if len(valid_actions) >= 3 and valid_actions[2]['amount']['min'] != -1:
                max_raise = valid_actions[2]['amount']['max']
                return 'raise', max_raise
            else:
                return 'call', call_amount
        else:
            if call_amount <= self.my_stack * 0.3:
                return 'call', call_amount
            else:
                return 'fold', 0
    
    def _improved_preflop_decision(self, valid_actions, hole_card, round_state, hand_strength):
        pot_size = self._get_pot_size(round_state)
        
        if self.position == 'SB':
            fold_threshold = 0.2
            call_threshold = 0.4
            raise_threshold = 0.55
            raise_frequency = 0.7
        else:
            fold_threshold = 0.3
            call_threshold = 0.5
            raise_threshold = 0.7
            raise_frequency = 0.65
        
        baseline_type = self.opponent_model['baseline_type']
        if baseline_type in self.baseline_adjustments:
            adj = self.baseline_adjustments[baseline_type]
            fold_threshold += adj.get('fold_threshold', 0)
            raise_frequency += adj.get('raise_frequency', 0)
        
        if self.game_phase == 'late':
            fold_threshold -= 0.1
            raise_frequency += 0.15
        
        if hand_strength < fold_threshold:
            return 'fold', 0
        elif hand_strength < call_threshold:
            if random.random() < 0.3:
                return self._size_raise(valid_actions, pot_size, 'small')
            return 'call', valid_actions[1]['amount']
        elif hand_strength < raise_threshold:
            if random.random() < raise_frequency:
                return self._size_raise(valid_actions, pot_size, 'medium')
            return 'call', valid_actions[1]['amount']
        else:
            if random.random() < 0.9:
                size_type = 'large' if hand_strength > 0.85 else 'medium'
                return self._size_raise(valid_actions, pot_size, size_type)
            return 'call', valid_actions[1]['amount']
    
    def _improved_postflop_decision(self, valid_actions, hole_card, round_state, hand_strength, pot_odds):
        pot_size = self._get_pot_size(round_state)
        call_amount = valid_actions[1]['amount']
        baseline_type = self.opponent_model['baseline_type']
        
        current_bluff_freq = self.bluff_frequency
        if baseline_type == 'tight':
            current_bluff_freq += 0.1
        elif baseline_type == 'loose':
            current_bluff_freq -= 0.05
        
        if hand_strength > 0.75:
            if random.random() < 0.8:
                return self._size_raise(valid_actions, pot_size, 'value')
            return 'call', call_amount
        
        elif hand_strength > 0.55:
            if pot_odds < 0.35:
                if baseline_type == 'passive' and random.random() < 0.5:
                    return self._size_raise(valid_actions, pot_size, 'small')
                return 'call', call_amount
            else:
                return 'fold', 0
        
        elif hand_strength > 0.35:
            if pot_odds < 0.25:
                return 'call', call_amount
            elif pot_odds < 0.4 and baseline_type == 'tight' and random.random() < current_bluff_freq:
                return self._size_raise(valid_actions, pot_size, 'bluff')
            else:
                return 'fold', 0
        
        else:
            if baseline_type == 'tight' and random.random() < current_bluff_freq:
                return self._size_raise(valid_actions, pot_size, 'bluff')
            elif pot_odds < 0.15:
                return 'call', call_amount
            else:
                return 'fold', 0
    
    def _size_raise(self, valid_actions, pot_size, raise_type):
        if len(valid_actions) < 3 or valid_actions[2]['amount']['min'] == -1:
            return 'call', valid_actions[1]['amount']
        
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']
        baseline_type = self.opponent_model['baseline_type']
        
        multipliers = {
            'small': 2.2,
            'medium': 3.0,
            'large': 4.2,
            'value': 2.8,
            'bluff': 2.5
        }
        
        base_multiplier = multipliers.get(raise_type, 2.5)
        
        if baseline_type == 'tight':
            base_multiplier *= 0.8
        elif baseline_type == 'loose':
            base_multiplier *= 1.2
        elif baseline_type == 'aggressive':
            base_multiplier *= 0.9
        
        if pot_size > 0:
            target_raise = min_raise + int(pot_size * base_multiplier)
        else:
            target_raise = min_raise * 2.5
        
        final_raise = max(min_raise, min(target_raise, max_raise))
        
        return 'raise', final_raise
    
    def _evaluate_hand_strength(self, hole_card, round_state):
        street = round_state['street']
        
        if street == 'preflop':
            return self._preflop_hand_strength(hole_card)
        else:
            return self._postflop_hand_strength(hole_card, round_state['community_card'])
    
    def _preflop_hand_strength(self, hole_card):
        if not hole_card or len(hole_card) != 2:
            return 0.1
        
        def parse_card(card_str):
            if len(card_str) != 2:
                return 2, 'C'
            suit = card_str[0]
            rank_char = card_str[1]
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                       '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            rank = rank_map.get(rank_char, 2)
            return rank, suit
        
        rank1, suit1 = parse_card(hole_card[0])
        rank2, suit2 = parse_card(hole_card[1])
        
        high_rank = max(rank1, rank2)
        low_rank = min(rank1, rank2)
        is_pair = rank1 == rank2
        is_suited = suit1 == suit2
        gap = high_rank - low_rank
        
        score = 0
        
        if is_pair and high_rank >= 8:
            score = 12 + high_rank
        elif high_rank == 14:
            score = 10 + low_rank
        elif high_rank == 13 and low_rank >= 8:
            score = 8 + low_rank
        elif high_rank == 12 and low_rank >= 9:
            score = 6 + low_rank
        elif high_rank == 11 and low_rank >= 10:
            score = 4 + high_rank + low_rank
        
        elif high_rank >= 11:
            score = 5 + low_rank
        elif is_suited and gap <= 2:
            score = 3 + high_rank + low_rank - gap
        elif gap <= 1 and high_rank >= 9:
            score = 2 + high_rank + low_rank
        
        elif high_rank >= 9:
            score = 1 + high_rank + low_rank
        elif is_suited:
            score = 1 + high_rank + low_rank
        else:
            score = max(0, high_rank + low_rank - 10)
        
        if is_suited:
            score += 2
        if is_pair:
            score += 3
        if gap == 0:
            score += 1
        
        if self.position == 'SB':
            score += 3
        
        if self.game_phase == 'late':
            score += 2
        
        normalized_score = min(max(score / 28.0, 0), 1)
        
        return normalized_score
    
    def _postflop_hand_strength(self, hole_card, community_card):
        if not community_card:
            return 0.3
        
        all_ranks = []
        all_suits = []
        
        try:
            for card in hole_card + community_card:
                if len(card) >= 2:
                    rank_char = card[1]
                    suit_char = card[0]
                    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                               '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                    rank = rank_map.get(rank_char, 2)
                    all_ranks.append(rank)
                    all_suits.append(suit_char)
        except Exception:
            return 0.3
        
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        trips = sum(1 for count in rank_counts.values() if count >= 3)
        quads = sum(1 for count in rank_counts.values() if count >= 4)
        
        base_strength = 0.2
        
        if quads > 0:
            base_strength = 0.98
        elif trips > 0:
            base_strength = 0.85
        elif pairs >= 2:
            base_strength = 0.7
        elif pairs >= 1:
            pair_rank = max(rank for rank, count in rank_counts.items() if count >= 2)
            base_strength = 0.4 + (pair_rank / 28.0)
        else:
            high_card_count = sum(1 for rank in all_ranks if rank >= 11)
            base_strength = 0.2 + (high_card_count * 0.08)
        
        suit_counts = {}
        for suit in all_suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= 5:
            base_strength = max(base_strength, 0.8)
        elif max_suit_count >= 4:
            base_strength += 0.1
        
        return min(base_strength, 1.0)
    
    def _update_game_state(self, round_state):
        try:
            seats = round_state.get('seats', [])
            if len(seats) >= 2:
                for i, seat in enumerate(seats):
                    if hasattr(self, 'uuid') and seat.get('uuid') == self.uuid:
                        self.my_stack = seat.get('stack', 1000)
                        small_blind_pos = round_state.get('small_blind_pos', 0)
                        if i == small_blind_pos:
                            self.position = 'SB'
                        else:
                            self.position = 'BB'
                    else:
                        self.opponent_stack = seat.get('stack', 1000)
            
            total_chips = self.my_stack + self.opponent_stack
            if self.my_stack < 250 or self.opponent_stack < 250:
                self.game_phase = 'late'
            elif total_chips < 1600:
                self.game_phase = 'middle'
            else:
                self.game_phase = 'early'
            
            self._update_opponent_model(round_state)
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _update_opponent_model(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            for street, actions in action_histories.items():
                if isinstance(actions, list):
                    for action in actions:
                        if isinstance(action, dict):
                            if hasattr(self, 'uuid') and action.get('uuid') != self.uuid:
                                action_type = action.get('action', '')
                                amount = action.get('amount', 0)
                                
                                self.opponent_model['recent_actions'].append({
                                    'action': action_type,
                                    'amount': amount,
                                    'street': street
                                })
                                
                                if len(self.opponent_model['recent_actions']) > 30:
                                    self.opponent_model['recent_actions'].pop(0)
        except Exception as e:
            print(f"Error updating opponent model: {e}")
    
    def _get_pot_size(self, round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = pot_info.get('side', [])
            side_pot_total = 0
            if isinstance(side_pots, list):
                side_pot_total = sum(side.get('amount', 0) for side in side_pots)
            return main_pot + side_pot_total
        except Exception:
            return 0
    
    def _safe_action(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            if call_amount <= self.my_stack * 0.15:
                return 'call', call_amount
            return 'fold', 0
        except Exception:
            return 'fold', 0
    
    def receive_game_start_message(self, game_info):
        try:
            self.my_stack = game_info['rule']['initial_stack']
            self.opponent_stack = game_info['rule']['initial_stack']
            self.round_count = 0
            self.game_phase = 'early'
            self.opponent_model['recent_actions'] = []
        except Exception as e:
            print(f"Error in receive_game_start_message: {e}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        try:
            self.round_count = round_count
            self.opponent_model['total_hands'] += 1
        except Exception as e:
            print(f"Error in receive_round_start_message: {e}")
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return RulePokerAI()