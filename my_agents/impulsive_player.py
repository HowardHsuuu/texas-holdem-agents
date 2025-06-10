from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
import numpy as np
from collections import defaultdict

class ImpulsivePokerAI(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        print("Initializing Aggressive Impulsive Poker AI...")
        
        self.aggression_level = 0.8
        self.bluff_frequency = 0.25
        self.tilt_factor = 0.0
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.games_won = 0
        self.games_lost = 0
        self.recent_results = []
        
        self.opponent_fold_rate = 0.5
        self.opponent_actions = {'fold': 0, 'call': 0, 'raise': 0}
        
        print("Aggressive AI ready for battle!")
        self._print_personality()
    
    def _print_personality(self):
        print("Aggressive AI Personality:")
        print(f"Base aggression: {self.aggression_level:.1f}")
        print(f"Bluff frequency: {self.bluff_frequency:.1f}")
        print(f"Current tilt: {self.tilt_factor:.1f}")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            
            current_aggression = self._calculate_current_aggression()
            
            hand_strength = self._aggressive_hand_evaluation(hole_card, round_state)
            
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            
            print(f"Aggro AI: Hand {hand_strength:.2f}, Aggression {current_aggression:.2f}, Pot {pot_size}, Call {call_amount}")
            
            decision = self._make_aggressive_decision(
                valid_actions, hand_strength, current_aggression, 
                pot_size, call_amount, round_state
            )
            
            self._track_our_action(decision[0])
            
            return decision
            
        except Exception as e:
            print(f"Aggro AI error: {e}")
            return self._emergency_aggressive_action(valid_actions)
    
    def _calculate_current_aggression(self):
        base_aggression = self.aggression_level
        
        tilt_bonus = self.tilt_factor * 0.3
        
        position_bonus = 0.1 if self.position == 'BB' else 0.0
        
        if self.my_stack + self.opponent_stack > 0:
            stack_ratio = self.my_stack / (self.my_stack + self.opponent_stack)
            stack_bonus = (stack_ratio - 0.5) * 0.2
        else:
            stack_bonus = 0.0
        
        late_game_bonus = 0.2 if self.round_count > 15 else 0.0
        
        total_aggression = base_aggression + tilt_bonus + position_bonus + stack_bonus + late_game_bonus
        return min(1.0, max(0.3, total_aggression))
    
    def _aggressive_hand_evaluation(self, hole_card, round_state):
        if not hole_card or len(hole_card) != 2:
            return 0.3
        
        try:
            strength = self._evaluate_with_engine(hole_card, round_state)
            if strength is not None:
                strength = min(1.0, max(0.1, strength))
            else:
                strength = self._basic_hand_strength(hole_card)
        except Exception as e:
            print(f"Hand evaluation error, using fallback: {e}")
            strength = self._basic_hand_strength(hole_card)
        
        community = round_state.get('community_card', [])
        
        if len(community) >= 3:
            draw_potential = self._evaluate_draws(hole_card, community)
            strength += draw_potential * 0.3
        
        if self.position == 'BB':
            strength += 0.1
        
        ranks = [self._get_rank(card) for card in hole_card]
        suits = [card[0] for card in hole_card]
        
        if suits[0] == suits[1] and abs(ranks[0] - ranks[1]) <= 4:
            strength += 0.15
        
        if 14 in ranks:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _evaluate_with_engine(self, hole_card, round_state):
        try:
            from game.engine.card import Card
            
            hole_cards = [Card.from_str(card) for card in hole_card]
            
            community_str = round_state.get('community_card', [])
            community_cards = [Card.from_str(card) for card in community_str]
            
            if len(hole_cards) == 2:
                hand_score = HandEvaluator.eval_hand(hole_cards, community_cards)
                
                normalized_strength = min(0.95, max(0.1, hand_score / 100000000.0))
                
                if normalized_strength < 0.05 or normalized_strength > 0.98:
                    return None
                
                return normalized_strength
            
        except Exception as e:
            print(f"Engine evaluation failed: {e}")
            return None
        
        return None
    
    def _basic_hand_strength(self, hole_card):
        ranks = [self._get_rank(card) for card in hole_card]
        suits = [card[0] for card in hole_card]
        
        strength = 0.0
        
        if ranks[0] == ranks[1]:
            if ranks[0] >= 12:
                strength = 0.9
            elif ranks[0] >= 9:
                strength = 0.75
            elif ranks[0] >= 6:
                strength = 0.6
            else:
                strength = 0.45
        else:
            high_rank = max(ranks)
            low_rank = min(ranks)
            
            if high_rank == 14:
                if low_rank >= 10:
                    strength = 0.8
                elif low_rank >= 7:
                    strength = 0.65
                else:
                    strength = 0.5
            elif high_rank >= 12:
                if low_rank >= 10:
                    strength = 0.7
                elif low_rank >= 8:
                    strength = 0.55
                else:
                    strength = 0.4
            elif high_rank >= 10:
                if low_rank >= 9:
                    strength = 0.6
                else:
                    strength = 0.35
            else:
                strength = 0.25
        
        if suits[0] == suits[1]:
            strength += 0.1
        
        if abs(ranks[0] - ranks[1]) <= 2:
            strength += 0.05
        
        return strength
    
    def _evaluate_draws(self, hole_card, community):
        all_cards = hole_card + community
        suits = [card[0] for card in all_cards]
        ranks = [self._get_rank(card) for card in all_cards]
        
        draw_strength = 0.0
        
        suit_counts = defaultdict(int)
        for suit in suits:
            suit_counts[suit] += 1
        
        max_suited = max(suit_counts.values()) if suit_counts else 0
        if max_suited == 4:
            draw_strength += 0.4
        elif max_suited == 3:
            draw_strength += 0.2
        
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 4:
            for i in range(len(unique_ranks) - 3):
                if unique_ranks[i+3] - unique_ranks[i] <= 4:
                    draw_strength += 0.3
                    break
        
        return draw_strength
    
    def _make_aggressive_decision(self, valid_actions, hand_strength, aggression, pot_size, call_amount, round_state):
        
        if hand_strength < 0.15 and call_amount > pot_size * 0.5:
            print("Even I'm not that crazy - folding trash")
            return 'fold', 0
        
        if random.random() < self.bluff_frequency * aggression:
            print("BLUFF TIME!")
            return self._make_bluff_raise(valid_actions, pot_size)
        
        if hand_strength > 0.65:
            print("Strong hand - ATTACK!")
            return self._make_value_raise(valid_actions, pot_size, 'strong')
        
        elif hand_strength > 0.45:
            if call_amount <= pot_size * 0.3:
                if random.random() < aggression:
                    print("Medium hand - PRESSURE!")
                    return self._make_value_raise(valid_actions, pot_size, 'medium')
                else:
                    print("Medium hand - call")
                    return 'call', call_amount
            else:
                print("Medium hand - expensive but calling")
                return 'call', call_amount
        
        elif hand_strength > 0.25:
            if call_amount <= 20 or call_amount <= pot_size * 0.2:
                print("Weak hand - cheap call")
                return 'call', call_amount
            else:
                if self.opponent_fold_rate > 0.6 and random.random() < 0.3:
                    print("Weak hand but opponent folds a lot - BLUFF!")
                    return self._make_bluff_raise(valid_actions, pot_size)
                else:
                    print("Weak hand - fold")
                    return 'fold', 0
        
        else:
            if call_amount <= 5:
                print("Trash but almost free")
                return 'call', call_amount
            else:
                print("Trash - fold")
                return 'fold', 0
    
    def _make_value_raise(self, valid_actions, pot_size, strength_level):
        if len(valid_actions) < 3:
            return 'call', valid_actions[1]['amount']
        
        raise_info = valid_actions[2]['amount']
        if not isinstance(raise_info, dict) or raise_info.get('min', -1) == -1:
            return 'call', valid_actions[1]['amount']
        
        min_raise = raise_info['min']
        max_raise = raise_info['max']
        
        if strength_level == 'strong':
            multiplier = random.uniform(0.8, 1.2)
        else:
            multiplier = random.uniform(0.5, 0.8)
        
        target_raise = min_raise + int(pot_size * multiplier)
        
        max_reasonable = min(self.my_stack * 0.4, max_raise)
        final_raise = max(min_raise, min(target_raise, max_reasonable))
        
        return 'raise', int(final_raise)
    
    def _make_bluff_raise(self, valid_actions, pot_size):
        if len(valid_actions) < 3:
            return 'call', valid_actions[1]['amount']
        
        raise_info = valid_actions[2]['amount']
        if not isinstance(raise_info, dict) or raise_info.get('min', -1) == -1:
            return 'call', valid_actions[1]['amount']
        
        min_raise = raise_info['min']
        max_raise = raise_info['max']
        
        multiplier = random.uniform(0.4, 0.7)
        target_raise = min_raise + int(pot_size * multiplier)
        
        if self.tilt_factor > 0.3:
            multiplier = random.uniform(0.8, 1.5)
            target_raise = min_raise + int(pot_size * multiplier)
        
        max_reasonable = min(self.my_stack * 0.3, max_raise)
        final_raise = max(min_raise, min(target_raise, max_reasonable))
        
        return 'raise', int(final_raise)
    
    def _emergency_aggressive_action(self, valid_actions):
        try:
            call_amount = valid_actions[1]['amount']
            if call_amount <= self.my_stack * 0.1:
                return 'call', call_amount
            return 'fold', 0
        except Exception as e:
            print(f"Emergency error: {e}")
            return 'fold', 0
    
    def _get_rank(self, card_str):
        if len(card_str) < 2:
            return 2
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(card_str[1], 2)
    
    def _get_pot_size(self, round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = sum(side.get('amount', 0) for side in pot_info.get('side', []))
            return main_pot + side_pots
        except Exception as e:
            print(f"Error getting pot size: {e}")
            return 50
    
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
            
            self._update_opponent_model(round_state)
            
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _update_opponent_model(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            total_actions = 0
            folds = 0
            
            for street_actions in action_histories.values():
                if isinstance(street_actions, list):
                    for action in street_actions:
                        if action.get('uuid') != self.uuid:
                            action_type = action.get('action', '').upper()
                            if action_type in ['FOLD', 'CALL', 'RAISE']:
                                total_actions += 1
                                action_lower = action_type.lower()
                                if action_lower in self.opponent_actions:
                                    self.opponent_actions[action_lower] += 1
                                if action_type == 'FOLD':
                                    folds += 1
            
            if total_actions > 0:
                self.opponent_fold_rate = folds / total_actions
                
        except Exception as e:
            print(f"Error updating opponent model: {e}")
    
    def _track_our_action(self, action):
        pass
    
    def _update_tilt(self, won):
        try:
            self.recent_results.append(won)
            if len(self.recent_results) > 5:
                self.recent_results.pop(0)
            
            recent_losses = sum(1 for result in self.recent_results if not result)
            if recent_losses >= 3:
                self.tilt_factor = min(0.5, self.tilt_factor + 0.1)
                print(f"Getting tilted! Tilt factor: {self.tilt_factor:.2f}")
            elif recent_losses == 0 and len(self.recent_results) >= 3:
                self.tilt_factor = max(0.0, self.tilt_factor - 0.1)
                print(f"Cooling down... Tilt factor: {self.tilt_factor:.2f}")
                
        except Exception as e:
            print(f"Error updating tilt: {e}")
    
    def receive_game_start_message(self, game_info):
        self.round_count = 0
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.opponent_actions = {'fold': 0, 'call': 0, 'raise': 0}
        print(f"Aggressive AI entering game #{self.games_won + self.games_lost + 1}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        try:
            for seat in seats:
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
        except Exception as e:
            print(f"Error in round start: {e}")
    
    def receive_street_start_message(self, street, round_state):
        if street == 'flop':
            print("Flop time - let's gamble!")
        elif street == 'turn':
            print("Turn card - decision time!")
        elif street == 'river':
            print("River - all or nothing!")
    
    def receive_game_update_message(self, action, round_state):
        try:
            player_uuid = action.get('player_uuid')
            if player_uuid != self.uuid:
                action_type = action.get('action', '').lower()
                amount = action.get('amount', 0)
                
                if action_type == 'raise' and amount > 100:
                    print(f"Opponent big raise {amount} - they're either strong or bluffing!")
                elif action_type == 'fold':
                    print("Opponent folded - pressure works!")
                    
        except Exception as e:
            print(f"Error processing game update: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            won = any(winner.get('uuid') == self.uuid for winner in winners)
            
            if won:
                self.games_won += 1
                print(f"VICTORY! Games won: {self.games_won}")
            else:
                self.games_lost += 1
                print(f"Defeat... Games lost: {self.games_lost}")
            
            self._update_tilt(won)
            
            if self.round_count % 10 == 0:
                total_games = self.games_won + self.games_lost
                win_rate = self.games_won / total_games if total_games > 0 else 0
                print(f"Aggressive AI Stats: {self.games_won}W-{self.games_lost}L ({win_rate:.1%} win rate)")
                print(f"Opponent fold rate: {self.opponent_fold_rate:.1%}")
                print(f"Current tilt: {self.tilt_factor:.2f}")
                
        except Exception as e:
            print(f"Error processing round result: {e}")

def setup_ai():
    return ImpulsivePokerAI()