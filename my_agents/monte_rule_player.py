from game.players import BasePokerPlayer
import random
import math

class MonteRulePokerAI(BasePokerPlayer):
    def __init__(self):
        self.opponent_model = {
            'vpip': 0.5,
            'pfr': 0.3,
            'aggression_factor': 1.0,
            'total_hands': 0,
            'fold_to_cbet': 0.6,
            'fold_to_3bet': 0.7,
            'cbet_frequency': 0.6,
            'recent_actions': [],
            'showdown_hands': [],
            'pattern_type': 'unknown',
            'confidence': 0.0,
            'all_in_frequency': 0.0,
            'bluff_frequency': 0.15,
            'bet_sizing_by_street': {
                'preflop': [],
                'flop': [],
                'turn': [],
                'river': []
            },
            'position_tendencies': {
                'button_open': 0.5,
                'bb_defend': 0.4
            },
            'exploitable_tells': []
        }
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.game_phase = 'early'
        self.recent_results = []
        self.desperation_factor = 1.0
        
        self.base_aggression = 1.5
        self.bluff_frequency = 0.25
        self.value_bet_sizing = 0.75
        self.meta_adjustments = {}
        
        self.mc_iterations = {
            'preflop': 0,
            'flop': 40,
            'turn': 60,
            'river': 80
        }
        
        self.base_all_in_thresholds = {
            'premium': 0.58,
            'strong': 0.62,
            'good': 0.66,
            'marginal': 0.72,
            'weak': 0.78
        }
        
        self.strategy_adjustments = {
            'aggression_modifier': 1.0,
            'bluff_modifier': 1.0,
            'call_threshold_modifier': 1.0
        }
        
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_comprehensive_game_state(round_state)
            
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            street = round_state['street']
            effective_stack = min(self.my_stack, self.opponent_stack)
            
            rule_strength = self._evaluate_rule_based_strength(hole_card, round_state)
            
            if call_amount >= self.my_stack * 0.75:
                return self._enhanced_all_in_decision(valid_actions, hole_card, rule_strength, round_state)
            
            mc_equity = None
            if street != 'preflop' and pot_size >= 40 and effective_stack > 100:
                try:
                    mc_equity = self._calculate_smart_monte_carlo(hole_card, round_state['community_card'], street)
                except:
                    mc_equity = None
            
            final_strength = self._advanced_strength_blending(rule_strength, mc_equity, street, pot_size, call_amount)
            
            final_strength = self._apply_exploitative_adjustments(final_strength, street, pot_size, effective_stack)
            
            final_strength = self._apply_meta_adjustments(final_strength, street)
            
            if street == 'preflop':
                return self._enhanced_preflop_decision(valid_actions, hole_card, final_strength, pot_size, effective_stack)
            else:
                return self._enhanced_postflop_decision(valid_actions, hole_card, round_state, final_strength, pot_size, effective_stack)
                
        except Exception as e:
            print(f"Error in declare_action: {e}")
            return self._enhanced_safe_action(valid_actions)
    
    def _enhanced_all_in_decision(self, valid_actions, hole_card, rule_strength, round_state):
        try:
            hand_category = self._classify_hand_strength(hole_card, round_state)
            base_threshold = self._get_dynamic_all_in_threshold(hand_category)
            
            my_ratio = self.my_stack / (self.my_stack + self.opponent_stack)
            opp_ratio = 1 - my_ratio
            
            if my_ratio < 0.25:
                base_threshold -= 0.12
            elif my_ratio < 0.35:
                base_threshold -= 0.08
            elif opp_ratio < 0.25:
                base_threshold += 0.06
            
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            
            if confidence > 0.7:
                if pattern == 'maniac':
                    if hand_category in ['premium', 'strong']:
                        base_threshold -= 0.10
                    elif hand_category == 'good':
                        base_threshold -= 0.06
                
                elif pattern == 'tight_aggressive':
                    base_threshold += 0.08
                
                elif pattern == 'calling_station':
                    base_threshold += 0.05
                
                elif pattern == 'loose_aggressive':
                    if hand_category in ['premium', 'strong']:
                        base_threshold -= 0.04
                    else:
                        base_threshold += 0.02
            
            if len(self.recent_results) >= 3:
                recent_losses = self.recent_results[-3:].count('loss')
                if recent_losses >= 2:
                    base_threshold -= 0.06
            
            blind_pressure = (self.round_count * 5) / max(self.my_stack, 1)
            if blind_pressure > 0.05:
                base_threshold -= min(blind_pressure * 2, 0.10)
            
            if rule_strength >= base_threshold:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']
        except Exception:
            return self._enhanced_safe_action(valid_actions)
    
    def _get_dynamic_all_in_threshold(self, hand_category):
        try:
            base = self.base_all_in_thresholds.get(hand_category, 0.7)
            
            if self.game_phase == 'late':
                base -= 0.08
            elif self.game_phase == 'middle':
                base -= 0.04
            
            base -= (self.desperation_factor - 1.0) * 0.15
            
            return max(base, 0.15)
        except Exception:
            return 0.6
    
    def _advanced_strength_blending(self, rule_strength, mc_equity, street, pot_size, call_amount):
        try:
            if mc_equity is None:
                return rule_strength
            
            pot_odds = call_amount / (pot_size + call_amount) if pot_size > 0 else 0
            
            if street == 'river':
                mc_weight = 0.65
            elif street == 'turn':
                mc_weight = 0.45
            else:
                mc_weight = 0.25
            
            decision_closeness = abs(rule_strength - pot_odds)
            if decision_closeness < 0.10:
                mc_weight += 0.15
            elif decision_closeness < 0.15:
                mc_weight += 0.08
            
            agreement = 1 - abs(rule_strength - mc_equity)
            if agreement > 0.85:
                mc_weight += 0.08
            
            if pot_size > 300:
                mc_weight += 0.12
            elif pot_size > 150:
                mc_weight += 0.06
            
            mc_weight = min(mc_weight, 0.75)
            
            if abs(rule_strength - mc_equity) > 0.25:
                mc_weight *= 0.7
            
            final_strength = (1 - mc_weight) * rule_strength + mc_weight * mc_equity
            return final_strength
        except Exception:
            return rule_strength
    
    def _apply_exploitative_adjustments(self, hand_strength, street, pot_size, effective_stack):
        try:
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            
            if confidence < 0.6:
                return hand_strength
            
            adjusted_strength = hand_strength
            
            if pattern == 'maniac':
                if hand_strength >= 0.65:
                    adjusted_strength = min(hand_strength + 0.06, 0.95)
                elif hand_strength <= 0.30:
                    adjusted_strength = max(hand_strength - 0.10, 0.05)
            
            elif pattern == 'tight_aggressive':
                if hand_strength <= 0.35:
                    adjusted_strength = min(hand_strength + 0.15, 0.50)
                elif 0.50 <= hand_strength <= 0.70:
                    adjusted_strength = min(hand_strength + 0.08, 0.80)
            
            elif pattern == 'calling_station':
                if hand_strength >= 0.45:
                    adjusted_strength = min(hand_strength + 0.12, 0.95)
                elif hand_strength <= 0.40:
                    adjusted_strength = max(hand_strength - 0.15, 0.05)
            
            elif pattern == 'loose_aggressive':
                if hand_strength >= 0.35:
                    adjusted_strength = min(hand_strength + 0.08, 0.90)
                elif hand_strength <= 0.25:
                    adjusted_strength = max(hand_strength - 0.05, 0.05)
            
            elif pattern == 'nit':
                if hand_strength <= 0.40:
                    adjusted_strength = min(hand_strength + 0.18, 0.60)
                elif 0.60 <= hand_strength <= 0.75:
                    adjusted_strength = max(hand_strength - 0.05, 0.55)
            
            for tell in self.opponent_model['exploitable_tells']:
                if tell == 'folds_to_large_bets':
                    if pot_size > 100:
                        adjusted_strength += 0.10
                elif tell == 'overvalues_pairs':
                    if street in ['turn', 'river']:
                        adjusted_strength += 0.05
            
            return adjusted_strength
        except Exception:
            return hand_strength
    
    def _apply_meta_adjustments(self, hand_strength, street):
        try:
            if len(self.recent_results) >= 4:
                recent_wins = self.recent_results[-4:].count('win')
                recent_losses = self.recent_results[-4:].count('loss')
                
                if recent_losses >= 3:
                    hand_strength *= 1.15
                    self.desperation_factor = min(self.desperation_factor + 0.1, 2.0)
                elif recent_wins >= 3:
                    hand_strength *= 0.95
                    self.desperation_factor = max(self.desperation_factor - 0.05, 0.8)
            
            my_ratio = self.my_stack / (self.my_stack + self.opponent_stack)
            blind_pressure = (self.round_count * 5) / max(self.my_stack, 1)
            
            if blind_pressure > 0.08:
                hand_strength *= (1 + blind_pressure * 1.5)
            
            if my_ratio < 0.3:
                hand_strength *= 1.2
            elif my_ratio > 0.7:
                hand_strength *= 1.1
            
            return min(hand_strength, 0.95)
        except Exception:
            return hand_strength
    
    def _enhanced_preflop_decision(self, valid_actions, hole_card, hand_strength, pot_size, effective_stack):
        try:
            position = self.position
            
            if position == 'SB':
                if effective_stack > 500:
                    fold_threshold = 0.12
                    call_threshold = 0.30
                    raise_threshold = 0.45
                    raise_frequency = 0.85
                elif effective_stack > 200:
                    fold_threshold = 0.15
                    call_threshold = 0.32
                    raise_threshold = 0.48
                    raise_frequency = 0.80
                else:
                    fold_threshold = 0.18
                    call_threshold = 0.35
                    raise_threshold = 0.55
                    raise_frequency = 0.75
            else:
                if effective_stack > 500:
                    fold_threshold = 0.20
                    call_threshold = 0.40
                    raise_threshold = 0.60
                    raise_frequency = 0.70
                elif effective_stack > 200:
                    fold_threshold = 0.25
                    call_threshold = 0.42
                    raise_threshold = 0.65
                    raise_frequency = 0.65
                else:
                    fold_threshold = 0.30
                    call_threshold = 0.45
                    raise_threshold = 0.70
                    raise_frequency = 0.60
            
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            
            if confidence > 0.6:
                if pattern == 'maniac':
                    fold_threshold += 0.08
                    raise_threshold += 0.05
                elif pattern == 'nit':
                    fold_threshold -= 0.06
                    raise_threshold -= 0.08
                    raise_frequency += 0.15
                elif pattern == 'calling_station':
                    raise_frequency += 0.10
                elif pattern == 'tight_aggressive':
                    fold_threshold += 0.03
                    raise_threshold += 0.03
            
            if self.desperation_factor > 1.2:
                fold_threshold -= 0.08
                raise_frequency += 0.15
            
            if hand_strength < fold_threshold:
                return valid_actions[0]['action'], valid_actions[0]['amount']
            
            elif hand_strength < call_threshold:
                if random.random() < 0.35:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'small', effective_stack)
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            elif hand_strength < raise_threshold:
                if random.random() < raise_frequency:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'medium', effective_stack)
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            else:
                if random.random() < 0.95:
                    size_type = 'large' if hand_strength > 0.85 else 'medium'
                    return self._enhanced_size_raise(valid_actions, pot_size, size_type, effective_stack)
                return valid_actions[1]['action'], valid_actions[1]['amount']
        except Exception:
            return self._enhanced_safe_action(valid_actions)
    
    def _enhanced_postflop_decision(self, valid_actions, hole_card, round_state, hand_strength, pot_size, effective_stack):
        try:
            call_amount = valid_actions[1]['amount']
            pot_odds = call_amount / (pot_size + call_amount) if pot_size > 0 else 0
            street = round_state['street']
            
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            
            if hand_strength > 0.80:
                if random.random() < 0.90:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'value', effective_stack)
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            elif hand_strength > 0.65:
                action_prob = 0.75
                
                if pattern == 'calling_station' and confidence > 0.6:
                    action_prob = 0.85
                elif pattern == 'tight_aggressive' and confidence > 0.6:
                    action_prob = 0.80
                elif pattern == 'maniac' and confidence > 0.6:
                    action_prob = 0.70
                
                if random.random() < action_prob:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'value', effective_stack)
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            elif hand_strength > 0.40:
                if hand_strength > pot_odds + 0.10:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
                
                bluff_chance = 0
                if pattern == 'nit' and confidence > 0.6:
                    bluff_chance = 0.30
                elif pattern == 'tight_aggressive' and confidence > 0.6:
                    bluff_chance = 0.20
                elif street == 'river' and self.position == 'SB':
                    bluff_chance = 0.15
                
                if random.random() < bluff_chance:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'bluff', effective_stack)
                
                if hand_strength < pot_odds + 0.05:
                    return valid_actions[0]['action'], valid_actions[0]['amount']
                
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            else:
                bluff_chance = 0
                
                if confidence > 0.6:
                    if pattern == 'nit':
                        bluff_chance = 0.35
                    elif pattern == 'tight_aggressive':
                        bluff_chance = 0.25
                    elif pattern == 'calling_station':
                        bluff_chance = 0.02
                    elif pattern == 'maniac':
                        bluff_chance = 0.05
                    else:
                        bluff_chance = 0.18
                else:
                    bluff_chance = 0.15
                
                if pot_odds < 0.25 and hand_strength > 0.20:
                    bluff_chance += 0.10
                
                if self.position == 'SB' and street in ['turn', 'river']:
                    bluff_chance += 0.08
                
                if random.random() < bluff_chance:
                    return self._enhanced_size_raise(valid_actions, pot_size, 'bluff', effective_stack)
                
                if pot_odds < 0.20 and hand_strength > 0.15:
                    return valid_actions[1]['action'], valid_actions[1]['amount']
                
                return valid_actions[0]['action'], valid_actions[0]['amount']
        except Exception:
            return self._enhanced_safe_action(valid_actions)
    
    def _enhanced_size_raise(self, valid_actions, pot_size, raise_type, effective_stack):
        try:
            if len(valid_actions) < 3 or valid_actions[2]['amount']['min'] == -1:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            
            min_raise = valid_actions[2]['amount']['min']
            max_raise = valid_actions[2]['amount']['max']
            
            multipliers = {
                'small': 2.2,
                'medium': 3.0,
                'large': 4.2,
                'value': 2.8,
                'bluff': 2.5
            }
            
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            base_multiplier = multipliers.get(raise_type, 2.8)
            
            if confidence > 0.6:
                if pattern == 'calling_station':
                    base_multiplier *= 1.4
                elif pattern == 'nit':
                    if raise_type == 'bluff':
                        base_multiplier *= 0.8
                    else:
                        base_multiplier *= 0.9
                elif pattern == 'maniac':
                    base_multiplier *= 1.1
                elif pattern == 'tight_aggressive':
                    base_multiplier *= 1.05
            
            stack_ratio = effective_stack / 1000
            
            if stack_ratio < 0.3:
                if raise_type in ['value', 'large']:
                    base_multiplier *= 1.3
            elif stack_ratio > 0.8:
                base_multiplier *= 0.9
            
            if pot_size > 0:
                target_raise = min_raise + int(pot_size * base_multiplier)
            else:
                target_raise = min_raise * 3
            
            variance = int(target_raise * 0.15)
            if variance > 0:
                target_raise += random.randint(-variance, variance)
            
            final_raise = max(min_raise, min(target_raise, max_raise))
            
            return 'raise', final_raise
        except Exception:
            return valid_actions[1]['action'], valid_actions[1]['amount']
    
    def _calculate_smart_monte_carlo(self, hole_cards, community_cards, street):
        
        if len(community_cards) < 3:
            return None
        
        iterations = self.mc_iterations.get(street, 40)
        
        try:
            wins = 0
            ties = 0
            valid_iterations = 0
            
            remaining_deck = self._get_remaining_deck(hole_cards + community_cards)
            
            for _ in range(iterations):
                try:
                    random.shuffle(remaining_deck)
                    if len(remaining_deck) < 2:
                        break
                    opp_hole = remaining_deck[:2]
                    
                    cards_needed = 5 - len(community_cards)
                    if cards_needed > 0:
                        available_cards = [card for card in remaining_deck[2:] if card not in opp_hole]
                        if len(available_cards) >= cards_needed:
                            final_board = community_cards + available_cards[:cards_needed]
                        else:
                            continue
                    else:
                        final_board = community_cards
                    
                    my_rank = self._evaluate_hand_rank(hole_cards + final_board)
                    opp_rank = self._evaluate_hand_rank(opp_hole + final_board)
                    
                    if my_rank > opp_rank:
                        wins += 1
                    elif my_rank == opp_rank:
                        ties += 1
                    
                    valid_iterations += 1
                    
                except:
                    continue
            
            if valid_iterations == 0:
                return None
            
            equity = (wins + ties * 0.5) / valid_iterations
            return equity
            
        except:
            return None
    
    def _update_comprehensive_game_state(self, round_state):
        try:
            self._update_basic_game_state(round_state)
            
            self._advanced_pattern_detection(round_state)
            
            self._detect_exploitable_tells(round_state)
            
            self._update_meta_factors()
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _update_basic_game_state(self, round_state):
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
            
            effective_stack = min(self.my_stack, self.opponent_stack)
            if effective_stack < 250:
                self.game_phase = 'late'
            elif effective_stack < 500:
                self.game_phase = 'middle'
            else:
                self.game_phase = 'early'
        except Exception:
            pass
    
    def _advanced_pattern_detection(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            opp_actions = []
            for street, actions in action_histories.items():
                if isinstance(actions, list):
                    for action in actions:
                        if hasattr(self, 'uuid') and action.get('uuid') != self.uuid:
                            opp_actions.append({
                                'action': action.get('action', ''),
                                'amount': action.get('amount', 0),
                                'street': street
                            })
            
            self.opponent_model['recent_actions'].extend(opp_actions)
            
            if len(self.opponent_model['recent_actions']) > 30:
                self.opponent_model['recent_actions'] = self.opponent_model['recent_actions'][-30:]
            
            if len(self.opponent_model['recent_actions']) >= 8:
                self._classify_opponent_pattern()
        except Exception:
            pass
    
    def _classify_opponent_pattern(self):
        try:
            actions = self.opponent_model['recent_actions']
            total_actions = len(actions)
            
            if total_actions < 8:
                return
            
            raises = sum(1 for a in actions if a['action'] == 'RAISE')
            calls = sum(1 for a in actions if a['action'] == 'CALL')
            folds = sum(1 for a in actions if a['action'] == 'FOLD')
            
            raise_freq = raises / total_actions
            call_freq = calls / total_actions
            fold_freq = folds / total_actions
            
            confidence = min(total_actions / 20.0, 1.0)
            
            if raise_freq > 0.6:
                self.opponent_model['pattern_type'] = 'maniac'
            elif fold_freq > 0.6:
                self.opponent_model['pattern_type'] = 'nit'
            elif call_freq > 0.55 and raise_freq < 0.20:
                self.opponent_model['pattern_type'] = 'calling_station'
            elif raise_freq > 0.35 and fold_freq > 0.30:
                self.opponent_model['pattern_type'] = 'tight_aggressive'
            elif raise_freq > 0.40 and call_freq > 0.35:
                self.opponent_model['pattern_type'] = 'loose_aggressive'
            elif 0.20 <= raise_freq <= 0.40 and 0.25 <= fold_freq <= 0.45:
                self.opponent_model['pattern_type'] = 'balanced'
            else:
                self.opponent_model['pattern_type'] = 'unknown'
                confidence *= 0.5
            
            self.opponent_model['confidence'] = confidence
        except Exception:
            pass
    
    def _detect_exploitable_tells(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            large_bets = 0
            folds_to_large = 0
            
            for street, actions in action_histories.items():
                if isinstance(actions, list):
                    pot_size = self._estimate_pot_size_at_street(street, action_histories)
                    
                    for action in actions:
                        if hasattr(self, 'uuid') and action.get('uuid') != self.uuid:
                            action_type = action.get('action', '')
                            amount = action.get('amount', 0)
                            
                            if action_type == 'RAISE' and pot_size > 0 and amount > pot_size * 0.75:
                                large_bets += 1
                            elif action_type == 'FOLD' and large_bets > 0:
                                folds_to_large += 1
            
            if large_bets > 0 and folds_to_large / large_bets > 0.7:
                if 'folds_to_large_bets' not in self.opponent_model['exploitable_tells']:
                    self.opponent_model['exploitable_tells'].append('folds_to_large_bets')
        except Exception:
            pass
    
    def _update_meta_factors(self):
        try:
            my_ratio = self.my_stack / (self.my_stack + self.opponent_stack)
            
            if my_ratio < 0.25:
                self.desperation_factor = min(self.desperation_factor + 0.1, 2.0)
            elif my_ratio > 0.75:
                self.desperation_factor = max(self.desperation_factor - 0.05, 0.8)
        except Exception:
            pass
    
    def _estimate_pot_size_at_street(self, street, action_histories):
        return 50
    
    def _evaluate_rule_based_strength(self, hole_card, round_state):
        try:
            street = round_state['street']
            
            if street == 'preflop':
                return self._enhanced_preflop_strength(hole_card)
            else:
                return self._enhanced_postflop_strength(hole_card, round_state['community_card'])
        except Exception:
            return 0.3
    
    def _enhanced_preflop_strength(self, hole_card):
        try:
            if not hole_card or len(hole_card) != 2:
                return 0.1
            
            card1, card2 = hole_card[0], hole_card[1]
            rank1 = self._card_rank_to_number(card1[1])
            rank2 = self._card_rank_to_number(card2[1])
            suit1, suit2 = card1[0], card2[0]
            
            high_rank = max(rank1, rank2)
            low_rank = min(rank1, rank2)
            is_pair = rank1 == rank2
            is_suited = suit1 == suit2
            gap = high_rank - low_rank
            
            base_strength = 0.1
            
            if is_pair and high_rank >= 13:
                base_strength = 0.88 + (high_rank - 13) * 0.04
            elif is_pair and high_rank == 12:
                base_strength = 0.84
            elif high_rank == 14 and low_rank == 13:
                base_strength = 0.82 + (0.04 if is_suited else 0)
            
            elif is_pair and high_rank >= 10:
                base_strength = 0.72 + (high_rank - 10) * 0.04
            elif high_rank == 14 and low_rank >= 11:
                base_strength = 0.74 + (low_rank - 11) * 0.03 + (0.05 if is_suited else 0)
            elif high_rank == 13 and low_rank == 12:
                base_strength = 0.70 + (0.05 if is_suited else 0)
            elif high_rank == 14 and low_rank == 10:
                base_strength = 0.68 + (0.06 if is_suited else 0)
            
            elif is_pair and high_rank >= 7:
                base_strength = 0.58 + (high_rank - 7) * 0.04
            elif high_rank == 14:
                base_strength = 0.48 + (low_rank - 2) * 0.025 + (0.08 if is_suited else 0)
            elif high_rank >= 12:
                base_strength = 0.42 + (high_rank + low_rank - 14) * 0.025 + (0.06 if is_suited else 0)
            
            elif is_suited and gap <= 1 and high_rank >= 8:
                base_strength = 0.38 + (high_rank + low_rank - 10) * 0.02
            elif is_suited and gap <= 2 and high_rank >= 9:
                base_strength = 0.28 + (high_rank + low_rank - 11) * 0.015
            elif gap <= 1 and high_rank >= 8:
                base_strength = 0.28 + (high_rank + low_rank - 10) * 0.015
            
            elif is_pair:
                base_strength = 0.38 + (high_rank - 2) * 0.025
            
            elif high_rank >= 12:
                base_strength = 0.22 + (high_rank + low_rank - 14) * 0.015
                if is_suited:
                    base_strength += 0.05
            
            else:
                base_strength = 0.16 + (high_rank + low_rank - 4) * 0.008
                if is_suited:
                    base_strength += 0.04
            
            if self.position == 'SB':
                base_strength += 0.10
            
            effective_stack = min(self.my_stack, self.opponent_stack)
            if effective_stack < 200:
                base_strength += 0.08
            elif effective_stack > 800:
                if is_suited or gap <= 2:
                    base_strength += 0.03
            
            if self.game_phase == 'late':
                base_strength += 0.08
            elif self.game_phase == 'middle':
                base_strength += 0.04
            
            pattern = self.opponent_model['pattern_type']
            confidence = self.opponent_model['confidence']
            
            if confidence > 0.6:
                if pattern == 'nit':
                    base_strength += 0.06
                elif pattern == 'maniac':
                    base_strength -= 0.03
                elif pattern == 'calling_station':
                    if base_strength > 0.5:
                        base_strength += 0.04
                    else:
                        base_strength -= 0.02
            
            return min(max(base_strength, 0.05), 0.95)
        except Exception:
            return 0.3
    
    def _enhanced_postflop_strength(self, hole_card, community_card):
        try:
            if not community_card:
                return 0.3
            
            all_cards = hole_card + community_card
            
            while len(all_cards) < 7:
                all_cards.append('S2')
            
            hand_rank = self._evaluate_hand_rank(all_cards)
            
            base_strength = self._rank_to_strength(hand_rank)
            
            if len(community_card) < 5:
                draw_strength = self._evaluate_drawing_potential(hole_card, community_card)
                base_strength += draw_strength
            
            return min(base_strength, 0.95)
        except Exception:
            return 0.3
    
    def _rank_to_strength(self, hand_rank):
        try:
            if hand_rank >= 8000:
                return 0.98
            elif hand_rank >= 7000:
                return 0.96
            elif hand_rank >= 6500:
                return 0.94
            elif hand_rank >= 6000:
                return 0.90
            elif hand_rank >= 5500:
                return 0.87
            elif hand_rank >= 5000:
                return 0.82
            elif hand_rank >= 4500:
                return 0.78
            elif hand_rank >= 4000:
                return 0.73
            elif hand_rank >= 3700:
                return 0.72
            elif hand_rank >= 3400:
                return 0.68
            elif hand_rank >= 3000:
                return 0.63
            elif hand_rank >= 2700:
                return 0.60
            elif hand_rank >= 2300:
                return 0.55
            elif hand_rank >= 2000:
                return 0.50
            elif hand_rank >= 1800:
                return 0.48
            elif hand_rank >= 1500:
                return 0.42
            elif hand_rank >= 1200:
                return 0.36
            elif hand_rank >= 1000:
                return 0.30
            elif hand_rank >= 800:
                return 0.22
            elif hand_rank >= 600:
                return 0.18
            else:
                return 0.15
        except Exception:
            return 0.3
    
    def _evaluate_drawing_potential(self, hole_card, community_card):
        try:
            if len(community_card) == 5:
                return 0
            
            all_cards = hole_card + community_card
            
            draw_value = 0
            
            suits = {}
            for card in all_cards:
                if len(card) >= 2:
                    suit = card[0]
                    suits[suit] = suits.get(suit, 0) + 1
            
            max_suit_count = max(suits.values()) if suits else 0
            
            if max_suit_count == 4:
                draw_value += 0.15
            elif max_suit_count == 3:
                draw_value += 0.03
            
            ranks = []
            for card in all_cards:
                if len(card) >= 2:
                    rank = self._card_rank_to_number(card[1])
                    ranks.append(rank)
            
            unique_ranks = sorted(set(ranks))
            
            for i in range(len(unique_ranks) - 3):
                if unique_ranks[i+3] - unique_ranks[i] == 3:
                    draw_value += 0.12
                    break
            
            for i in range(len(unique_ranks) - 2):
                if unique_ranks[i+2] - unique_ranks[i] == 3:
                    draw_value += 0.06
                    break
            
            return min(draw_value, 0.25)
        except Exception:
            return 0
    
    def _classify_hand_strength(self, hole_card, round_state):
        try:
            if not hole_card or len(hole_card) != 2:
                return 'weak'
            
            street = round_state['street']
            
            if street == 'preflop':
                return self._classify_preflop_hand_enhanced(hole_card)
            else:
                return self._classify_postflop_hand_enhanced(hole_card, round_state['community_card'])
        except Exception:
            return 'weak'
    
    def _classify_preflop_hand_enhanced(self, hole_card):
        try:
            card1, card2 = hole_card[0], hole_card[1]
            rank1 = self._card_rank_to_number(card1[1])
            rank2 = self._card_rank_to_number(card2[1])
            suit1, suit2 = card1[0], card2[0]
            
            high_rank = max(rank1, rank2)
            low_rank = min(rank1, rank2)
            is_pair = rank1 == rank2
            is_suited = suit1 == suit2
            gap = high_rank - low_rank
            
            if (is_pair and high_rank >= 13) or (high_rank == 14 and low_rank == 13):
                return 'premium'
            
            if (is_pair and high_rank >= 10) or \
               (high_rank == 14 and low_rank >= 11) or \
               (high_rank == 13 and low_rank >= 11 and is_suited) or \
               (is_pair and high_rank == 9):
                return 'strong'
            
            if (is_pair and high_rank >= 7) or \
               (high_rank == 14 and low_rank >= 9) or \
               (high_rank >= 12 and low_rank >= 10) or \
               (is_suited and gap <= 1 and high_rank >= 9) or \
               (is_suited and high_rank == 14 and low_rank >= 7):
                return 'good'
            
            if (is_pair) or \
               (high_rank == 14) or \
               (high_rank >= 11) or \
               (is_suited and gap <= 2 and high_rank >= 8) or \
               (gap <= 1 and high_rank >= 9):
                return 'marginal'
            
            return 'weak'
        except Exception:
            return 'weak'
    
    def _classify_postflop_hand_enhanced(self, hole_card, community_card):
        try:
            all_cards = hole_card + community_card
            hand_rank = self._evaluate_hand_rank(all_cards)
            
            if hand_rank >= 7000:
                return 'premium'
            elif hand_rank >= 6000:
                return 'premium'
            elif hand_rank >= 5000:
                return 'strong'
            elif hand_rank >= 4000:
                return 'strong'
            elif hand_rank >= 3000:
                return 'strong'
            elif hand_rank >= 2500:
                return 'good'
            elif hand_rank >= 2000:
                return 'good'
            elif hand_rank >= 1500:
                return 'good'
            elif hand_rank >= 1000:
                return 'marginal'
            else:
                return 'weak'
        except Exception:
            return 'weak'
    
    def _get_remaining_deck(self, known_cards):
        try:
            full_deck = self._create_full_deck()
            known_set = set(known_cards)
            return [card for card in full_deck if card not in known_set]
        except Exception:
            return []
    
    def _create_full_deck(self):
        suits = ['S', 'H', 'D', 'C']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [suit + rank for suit in suits for rank in ranks]
    
    def _evaluate_hand_rank(self, seven_cards):
        try:
            if len(seven_cards) != 7:
                return 0
            
            ranks = []
            suits = []
            
            for card in seven_cards:
                if len(card) >= 2:
                    rank = self._card_rank_to_number(card[1])
                    suit = card[0]
                    ranks.append(rank)
                    suits.append(suit)
            
            rank_counts = {}
            suit_counts = {}
            
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            has_flush = max(suit_counts.values()) >= 5
            straight_high = self._check_straight_high(ranks)
            has_straight = straight_high > 0
            
            if has_flush and has_straight:
                return 8000 + straight_high
            
            quads = [rank for rank, count in rank_counts.items() if count == 4]
            if quads:
                quad_rank = max(quads)
                kicker = max([r for r in ranks if r != quad_rank])
                return 7000 + quad_rank * 15 + kicker
            
            trips = [rank for rank, count in rank_counts.items() if count >= 3]
            pairs = [rank for rank, count in rank_counts.items() if count >= 2 and rank not in trips]
            
            if trips:
                trip_rank = max(trips)
                if len(trips) > 1:
                    pair_rank = sorted(trips)[-2]
                elif pairs:
                    pair_rank = max(pairs)
                else:
                    pair_rank = 0
                
                if pair_rank > 0:
                    return 6000 + trip_rank * 15 + pair_rank
            
            if has_flush:
                flush_suit = max(suit_counts, key=suit_counts.get)
                flush_ranks = sorted([rank for rank, suit in zip(ranks, suits) if suit == flush_suit], reverse=True)[:5]
                return 5000 + sum(rank * (0.01 ** i) for i, rank in enumerate(flush_ranks)) * 100
            
            if has_straight:
                return 4000 + straight_high
            
            if trips:
                trip_rank = max(trips)
                kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)[:2]
                return 3000 + trip_rank * 20 + sum(k * (0.1 ** (i+1)) for i, k in enumerate(kickers)) * 100
            
            if len(pairs) >= 2:
                pair_ranks = sorted(pairs, reverse=True)[:2]
                kicker = max([r for r in ranks if r not in pair_ranks])
                return 2000 + pair_ranks[0] * 20 + pair_ranks[1] + kicker * 0.1
            
            elif pairs:
                pair_rank = max(pairs)
                kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)[:3]
                return 1000 + pair_rank * 20 + sum(k * (0.1 ** (i+1)) for i, k in enumerate(kickers)) * 100
            
            else:
                high_cards = sorted(ranks, reverse=True)[:5]
                return sum(card * (0.1 ** i) for i, card in enumerate(high_cards)) * 10
        except Exception:
            return 0
    
    def _check_straight_high(self, ranks):
        try:
            unique_ranks = sorted(set(ranks))
            
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    return unique_ranks[i+4]
            
            if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
                return 5
            
            return 0
        except Exception:
            return 0
    
    def _card_rank_to_number(self, rank_char):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank_char, 2)
    
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
    
    def _enhanced_safe_action(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            
            if call_amount <= self.my_stack * 0.15:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            elif call_amount <= self.my_stack * 0.05:
                return valid_actions[1]['action'], valid_actions[1]['amount']
            else:
                return valid_actions[0]['action'], valid_actions[0]['amount']
        except Exception:
            return 'fold', 0
    
    def receive_game_start_message(self, game_info):
        try:
            self.opponent_model = {
                'vpip': 0.5, 'pfr': 0.3, 'aggression_factor': 1.0, 'total_hands': 0,
                'fold_to_cbet': 0.6, 'fold_to_3bet': 0.7, 'cbet_frequency': 0.6,
                'recent_actions': [], 'showdown_hands': [], 'pattern_type': 'unknown',
                'confidence': 0.0, 'all_in_frequency': 0.0, 'bluff_frequency': 0.15,
                'bet_sizing_by_street': {'preflop': [], 'flop': [], 'turn': [], 'river': []},
                'position_tendencies': {'button_open': 0.5, 'bb_defend': 0.4},
                'exploitable_tells': []
            }
            
            self.my_stack = game_info['rule']['initial_stack']
            self.opponent_stack = game_info['rule']['initial_stack']
            self.round_count = 0
            self.game_phase = 'early'
            self.recent_results = []
            self.desperation_factor = 1.0
            
            self.strategy_adjustments = {
                'aggression_modifier': 1.0,
                'bluff_modifier': 1.0,
                'call_threshold_modifier': 1.0
            }
        except Exception as e:
            print(f"Error in receive_game_start_message: {e}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        try:
            self.round_count = round_count
            self.opponent_model['total_hands'] += 1
            
            if round_count > 15:
                self.base_aggression = 1.7
                self.bluff_frequency = 0.32
                self.mc_iterations = {'preflop': 0, 'flop': 30, 'turn': 50, 'river': 70}
                
                for hand_type in self.base_all_in_thresholds:
                    self.base_all_in_thresholds[hand_type] -= 0.06
                    
            elif round_count > 10:
                self.base_aggression = 1.6
                self.bluff_frequency = 0.28
                self.mc_iterations = {'preflop': 0, 'flop': 35, 'turn': 55, 'river': 75}
                
                for hand_type in self.base_all_in_thresholds:
                    self.base_all_in_thresholds[hand_type] -= 0.03
            
            elif round_count > 5:
                self.base_aggression = 1.5
                self.bluff_frequency = 0.26
        except Exception as e:
            print(f"Error in receive_round_start_message: {e}")
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        try:
            if hasattr(self, 'uuid') and action.get('player_uuid') != self.uuid:
                street = round_state.get('street', 'unknown')
                pot_size = self._get_pot_size(round_state)
                action_amount = action.get('amount', 0)
                
                action_data = {
                    'action': action.get('action'),
                    'amount': action_amount,
                    'street': street,
                    'pot_size': pot_size,
                    'bet_ratio': (action_amount / pot_size) if pot_size > 0 else 0,
                    'round': self.round_count
                }
                
                self.opponent_model['recent_actions'].append(action_data)
                
                if action.get('action') == 'RAISE' and pot_size > 0:
                    bet_ratio = action_amount / pot_size
                    self.opponent_model['bet_sizing_by_street'][street].append(bet_ratio)
                    
                    if len(self.opponent_model['bet_sizing_by_street'][street]) > 10:
                        self.opponent_model['bet_sizing_by_street'][street].pop(0)
                
                if len(self.opponent_model['recent_actions']) > 25:
                    self.opponent_model['recent_actions'] = self.opponent_model['recent_actions'][-25:]
        except Exception as e:
            print(f"Error in receive_game_update_message: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            my_won = any(winner.get('uuid') == self.uuid for winner in winners)
            self.recent_results.append('win' if my_won else 'loss')
            
            if len(self.recent_results) > 8:
                self.recent_results.pop(0)
            
            for info in hand_info:
                if hasattr(self, 'uuid') and info.get('uuid') != self.uuid:
                    hand_strength_name = info.get('hand', {}).get('strength', '')
                    self.opponent_model['showdown_hands'].append({
                        'strength': hand_strength_name,
                        'round': self.round_count,
                        'was_aggressive': self._was_opponent_aggressive_this_hand(round_state)
                    })
                    
                    if self._was_opponent_aggressive_this_hand(round_state):
                        if hand_strength_name in ['HIGHCARD', 'ONEPAIR']:
                            self.opponent_model['bluff_frequency'] = min(
                                self.opponent_model['bluff_frequency'] + 0.05, 0.4
                            )
                        else:
                            self.opponent_model['bluff_frequency'] = max(
                                self.opponent_model['bluff_frequency'] - 0.02, 0.05
                            )
            
            if len(self.opponent_model['showdown_hands']) > 6:
                self.opponent_model['showdown_hands'] = self.opponent_model['showdown_hands'][-6:]
        except Exception as e:
            print(f"Error in receive_round_result_message: {e}")
    
    def _was_opponent_aggressive_this_hand(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            for street, actions in action_histories.items():
                if isinstance(actions, list):
                    for action in actions:
                        if (hasattr(self, 'uuid') and 
                            action.get('uuid') != self.uuid and 
                            action.get('action') == 'RAISE'):
                            return True
            
            return False
        except Exception:
            return False

def setup_ai():
    return MonteRulePokerAI()