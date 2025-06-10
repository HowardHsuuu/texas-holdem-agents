import random
import math
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator

class ProbabilityAgent(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        self.name = "ImprovedPB"
        
        self.mu = 3.8
        self.sigma = 18.0
        
        self.opponent_aggression = 0.5
        self.opponent_skill_level = "medium"
        self.games_played = 0
        self.recent_losses = 0
        
        self.risk_tolerance = 1.0
        
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            current_stack = self._get_my_stack(round_state)
            remaining_rounds = self._get_remaining_rounds(round_state)
            current_profit = current_stack - 1000
            pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
            
            self._assess_opponent_strength(round_state)
            
            round_rate = self._calculate_enhanced_round_rate(hole_card, round_state)
            
            risk_factor = self._calculate_risk_factor(current_profit, remaining_rounds)
            
            fold_ev = self._calculate_fold_ev_enhanced(current_profit, remaining_rounds)
            call_ev = self._calculate_call_ev_enhanced(current_profit, remaining_rounds, valid_actions, round_rate, pot_size, risk_factor)
            raise_ev, optimal_raise = self._calculate_raise_ev_enhanced(current_profit, remaining_rounds, valid_actions, round_rate, pot_size, risk_factor)
            
            if fold_ev >= call_ev and fold_ev >= raise_ev:
                return "fold", 0
            elif call_ev >= raise_ev:
                return "call", valid_actions[1]["amount"]
            else:
                min_raise = valid_actions[2]["amount"]["min"]
                max_raise = valid_actions[2]["amount"]["max"]
                if min_raise == -1:
                    return "call", valid_actions[1]["amount"]
                raise_amount = max(min_raise, min(optimal_raise, max_raise))
                return "raise", raise_amount
                
        except Exception:
            return "call", valid_actions[1]["amount"]
    
    def _calculate_enhanced_round_rate(self, hole_card, round_state, simulations=800):
        try:
            community_cards = round_state.get('community_card', [])
            
            if len(community_cards) == 0:
                return self._enhanced_preflop_rate(hole_card)
            
            hole_cards = [Card.from_str(card) for card in hole_card]
            community = [Card.from_str(card) for card in community_cards]
            
            all_cards = [Card.from_id(i) for i in range(1, 53)]
            used_cards = set(card.to_id() for card in hole_cards + community)
            available_cards = [card for card in all_cards if card.to_id() not in used_cards]
            
            wins = 0
            total_sims = 0
            
            for _ in range(simulations):
                try:
                    deck = available_cards.copy()
                    random.shuffle(deck)
                    
                    opp_hole = self._select_opponent_hand(deck)
                    
                    sim_community = community.copy()
                    cards_needed = 5 - len(sim_community)
                    if cards_needed > 0:
                        sim_community.extend(deck[2:2 + cards_needed])
                    
                    my_strength = HandEvaluator.eval_hand(hole_cards, sim_community)
                    opp_strength = HandEvaluator.eval_hand(opp_hole, sim_community)
                    
                    if my_strength > opp_strength:
                        wins += 1
                    elif my_strength == opp_strength:
                        wins += 0.5
                    
                    total_sims += 1
                    
                except Exception:
                    continue
            
            if total_sims == 0:
                return self._enhanced_preflop_rate(hole_card)
            
            base_rate = wins / total_sims
            
            if self.opponent_skill_level == "strong":
                base_rate *= 0.92
            elif self.opponent_skill_level == "weak":
                base_rate *= 1.08
            
            return min(0.95, max(0.05, base_rate))
            
        except Exception:
            return 0.5
    
    def _enhanced_preflop_rate(self, hole_card):
        try:
            if len(hole_card) != 2:
                return 0.5
            
            card1, card2 = hole_card
            rank1 = self._parse_rank(card1[1])
            rank2 = self._parse_rank(card2[1])
            suit1, suit2 = card1[0], card2[0]
            
            high_rank = max(rank1, rank2)
            low_rank = min(rank1, rank2)
            is_pair = rank1 == rank2
            is_suited = suit1 == suit2
            
            if is_pair:
                pair_rates = {14: 0.87, 13: 0.84, 12: 0.78, 11: 0.75, 10: 0.72,
                             9: 0.68, 8: 0.65, 7: 0.62, 6: 0.58, 5: 0.55,
                             4: 0.52, 3: 0.50, 2: 0.48}
                base_rate = pair_rates.get(high_rank, 0.5)
            else:
                if high_rank == 14:
                    if low_rank >= 13: base_rate = 0.75
                    elif low_rank >= 12: base_rate = 0.68
                    elif low_rank >= 11: base_rate = 0.64
                    elif low_rank >= 10: base_rate = 0.60
                    elif low_rank >= 9: base_rate = 0.55
                    else: base_rate = 0.48
                elif high_rank == 13:
                    if low_rank >= 12: base_rate = 0.63
                    elif low_rank >= 11: base_rate = 0.58
                    elif low_rank >= 10: base_rate = 0.54
                    else: base_rate = 0.45
                elif high_rank >= 11:
                    if low_rank >= 10: base_rate = 0.56
                    elif low_rank >= 9: base_rate = 0.52
                    else: base_rate = 0.42
                else:
                    base_rate = 0.35 + (high_rank - 2) * 0.02
                
                if is_suited:
                    base_rate += 0.06
                
                gap = abs(high_rank - low_rank)
                if gap <= 1: base_rate += 0.04
                elif gap <= 3: base_rate += 0.02
            
            if self.opponent_skill_level == "strong":
                base_rate *= 0.94
            elif self.opponent_skill_level == "weak":
                base_rate *= 1.06
            
            return min(0.90, max(0.25, base_rate))
            
        except Exception:
            return 0.5
    
    def _parse_rank(self, rank_char):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank_char, 2)
    
    def _select_opponent_hand(self, deck):
        return deck[:2]
    
    def _calculate_risk_factor(self, current_profit, remaining_rounds):
        try:
            risk_factor = self.risk_tolerance
            
            if current_profit < -200:
                risk_factor *= 0.8
            elif current_profit > 200:
                risk_factor *= 1.2
            
            if remaining_rounds <= 5:
                if current_profit < 0:
                    risk_factor *= 1.3
                else:
                    risk_factor *= 0.9
            
            if self.opponent_skill_level == "strong":
                risk_factor *= 0.85
            
            return max(0.5, min(1.5, risk_factor))
            
        except Exception:
            return 1.0
    
    def _calculate_call_ev_enhanced(self, current_profit, remaining_rounds, valid_actions, round_rate, pot_size, risk_factor):
        try:
            call_amount = valid_actions[1]["amount"]
            
            if call_amount > 0 and pot_size > 0:
                pot_odds = pot_size / call_amount
                required_equity = 1 / (1 + pot_odds)
                
                if round_rate > required_equity * 0.9:
                    pot_odds_bonus = 0.1 * risk_factor
                else:
                    pot_odds_bonus = -0.05
            else:
                pot_odds_bonus = 0
            
            win_profit = current_profit + pot_size + call_amount
            lose_profit = current_profit - call_amount
            
            win_game_rate = self._calculate_game_rate(win_profit, remaining_rounds - 1)
            lose_game_rate = self._calculate_game_rate(lose_profit, remaining_rounds - 1)
            
            base_ev = round_rate * win_game_rate + (1 - round_rate) * lose_game_rate
            
            return base_ev + pot_odds_bonus
            
        except Exception:
            return 0.0
    
    def _calculate_raise_ev_enhanced(self, current_profit, remaining_rounds, valid_actions, round_rate, pot_size, risk_factor):
        try:
            min_raise = valid_actions[2]["amount"]["min"]
            max_raise = valid_actions[2]["amount"]["max"]
            
            if min_raise == -1:
                return 0.0, 0
            
            best_ev = 0.0
            best_raise = min_raise
            
            if self.opponent_skill_level == "strong":
                test_multipliers = [1.0, 1.5, 2.0]
            else:
                test_multipliers = [1.0, 2.0, 3.0, 5.0]
            
            for multiplier in test_multipliers:
                raise_amount = min(min_raise * multiplier, max_raise)
                if raise_amount < min_raise:
                    continue
                
                fold_equity = self._estimate_fold_equity_enhanced(raise_amount, pot_size)
                
                fold_ev = self._calculate_game_rate(current_profit + pot_size, remaining_rounds - 1)
                
                win_profit = current_profit + pot_size + raise_amount
                lose_profit = current_profit - raise_amount
                
                win_game_rate = self._calculate_game_rate(win_profit, remaining_rounds - 1)
                lose_game_rate = self._calculate_game_rate(lose_profit, remaining_rounds - 1)
                
                call_ev = round_rate * win_game_rate + (1 - round_rate) * lose_game_rate
                
                total_ev = fold_equity * fold_ev + (1 - fold_equity) * call_ev
                total_ev *= risk_factor
                
                if total_ev > best_ev:
                    best_ev = total_ev
                    best_raise = raise_amount
            
            return best_ev, best_raise
            
        except Exception:
            return 0.0, min_raise if min_raise != -1 else 0
    
    def _estimate_fold_equity_enhanced(self, raise_amount, pot_size):
        try:
            if pot_size == 0:
                base_fold_rate = 0.25
            else:
                bet_to_pot = raise_amount / pot_size
                base_fold_rate = min(0.7, 0.15 + bet_to_pot * 0.25)
            
            if self.opponent_skill_level == "strong":
                base_fold_rate *= 0.8
            elif self.opponent_skill_level == "weak":
                base_fold_rate *= 1.2
            
            aggression_adjustment = (0.5 - self.opponent_aggression) * 0.2
            
            return max(0.05, min(0.8, base_fold_rate + aggression_adjustment))
            
        except Exception:
            return 0.3
    
    def _calculate_fold_ev_enhanced(self, current_profit, remaining_rounds):
        return self._calculate_game_rate(current_profit, remaining_rounds - 1)
    
    def _assess_opponent_strength(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            
            total_actions = 0
            raises = 0
            calls = 0
            folds = 0
            
            for street, actions in action_histories.items():
                for action in actions:
                    if action.get('uuid') != self.uuid:
                        total_actions += 1
                        action_type = action.get('action', '')
                        if action_type == 'RAISE':
                            raises += 1
                        elif action_type == 'CALL':
                            calls += 1
                        elif action_type == 'FOLD':
                            folds += 1
            
            if total_actions > 5:
                self.opponent_aggression = raises / total_actions if total_actions > 0 else 0.5
                
                if self.opponent_aggression > 0.4:
                    self.opponent_skill_level = "strong"
                elif self.opponent_aggression < 0.2:
                    self.opponent_skill_level = "weak"
                else:
                    self.opponent_skill_level = "medium"
                    
        except Exception:
            pass
    
    def _calculate_game_rate(self, current_profit, remaining_rounds):
        try:
            if remaining_rounds <= 0:
                return 1.0 if current_profit > 0 else 0.0
            
            expected_profit = remaining_rounds * self.mu
            variance = remaining_rounds * (self.sigma ** 2)
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                return 1.0 if expected_profit > -current_profit else 0.0
            
            z = (expected_profit + current_profit) / std_dev
            return self._normal_cdf(z)
            
        except Exception:
            return 0.5
    
    def _normal_cdf(self, z):
        try:
            if z < -6: return 0.0
            if z > 6: return 1.0
            
            t = 1.0 / (1.0 + 0.2316419 * abs(z))
            d = 0.3989423 * math.exp(-z * z / 2.0)
            p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
            
            return 1.0 - p if z >= 0 else p
        except Exception:
            return 0.5
    
    def _get_my_stack(self, round_state):
        try:
            for seat in round_state['seats']:
                if seat['uuid'] == self.uuid:
                    return seat['stack']
        except Exception:
            pass
        return 1000
    
    def _get_remaining_rounds(self, round_state):
        try:
            current_round = round_state.get('round_count', 1)
            return max(1, 20 - current_round + 1)
        except Exception:
            return 10
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            if any(winner.get('uuid') == self.uuid for winner in winners):
                self.recent_losses = max(0, self.recent_losses - 1)
            else:
                self.recent_losses += 1
                
            if self.recent_losses >= 3:
                self.risk_tolerance *= 0.9
                self.mu *= 0.95
            elif self.recent_losses == 0:
                self.risk_tolerance = min(1.2, self.risk_tolerance * 1.05)
                
        except Exception:
            pass


def setup_ai():
    return ProbabilityAgent()