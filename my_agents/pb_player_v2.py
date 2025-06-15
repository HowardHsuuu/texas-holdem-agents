import random
import math
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator

class PBV2PokerAgent(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        self.name = "PB-v2"
        
        self.mu = 4.2
        self.sigma = 16.5
        
        self.opponent_actions = []
        self.total_opponent_actions = 0
        self.opponent_raises = 0
        self.opponent_calls = 0
        self.opponent_folds = 0
        
        self.games_played = 0
        self.recent_performance = []
        
        self.base_aggression = 1.0
        
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            current_stack = self._get_my_stack(round_state)
            remaining_rounds = self._get_remaining_rounds(round_state)
            current_profit = current_stack - 1000
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]["amount"]
            
            position = self._get_position(round_state)
            
            win_prob = self._calculate_win_probability(hole_card, round_state)
            
            fold_ev = self._calculate_fold_ev(current_profit, remaining_rounds)
            call_ev = self._calculate_call_ev(current_profit, remaining_rounds, call_amount, pot_size, win_prob, position)
            raise_ev, optimal_raise = self._calculate_raise_ev(current_profit, remaining_rounds, valid_actions, pot_size, win_prob, position)
            
            if fold_ev >= call_ev and fold_ev >= raise_ev:
                return "fold", 0
            elif call_ev >= raise_ev:
                return "call", call_amount
            else:
                min_raise = valid_actions[2]["amount"]["min"]
                max_raise = valid_actions[2]["amount"]["max"]
                if min_raise == -1:
                    return "call", call_amount
                raise_amount = max(min_raise, min(optimal_raise, max_raise))
                return "raise", raise_amount
                
        except Exception as e:
            return "call", valid_actions[1]["amount"]
    
    def _calculate_win_probability(self, hole_card, round_state):
        try:
            community_cards = round_state.get('community_card', [])
            
            if len(community_cards) == 0:
                return self._evaluate_preflop_strength(hole_card)
            
            return self._monte_carlo_simulation(hole_card, community_cards)
            
        except Exception:
            return 0.5
    
    def _evaluate_preflop_strength(self, hole_card):
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
            gap = high_rank - low_rank
            
            if is_pair:
                if high_rank >= 13:
                    return 0.85
                elif high_rank >= 10:
                    return 0.75
                elif high_rank >= 8:
                    return 0.65
                elif high_rank >= 6:
                    return 0.58
                else:
                    return 0.52
            
            if high_rank == 14:
                if low_rank >= 13:
                    return 0.75 if is_suited else 0.70
                elif low_rank >= 12:
                    return 0.68 if is_suited else 0.62
                elif low_rank >= 11:
                    return 0.63 if is_suited else 0.58
                elif low_rank >= 10:
                    return 0.60 if is_suited else 0.55
                elif low_rank >= 9:
                    return 0.57 if is_suited else 0.50
                elif low_rank >= 7:
                    return 0.54 if is_suited else 0.45
                else:
                    return 0.50 if is_suited else 0.40
            
            elif high_rank == 13:
                if low_rank >= 12:
                    return 0.65 if is_suited else 0.60
                elif low_rank >= 11:
                    return 0.60 if is_suited else 0.55
                elif low_rank >= 10:
                    return 0.58 if is_suited else 0.52
                elif low_rank >= 9:
                    return 0.54 if is_suited else 0.47
                else:
                    return 0.48 if is_suited else 0.40
            
            elif high_rank >= 11:
                if gap <= 1:
                    return 0.58 if is_suited else 0.52
                elif gap <= 3:
                    return 0.52 if is_suited else 0.45
                else:
                    return 0.45 if is_suited else 0.38
            
            else:
                if gap == 0:
                    return 0.55 if is_suited else 0.48
                elif gap <= 2 and is_suited:
                    return 0.50
                else:
                    return 0.35
            
        except Exception:
            return 0.5
    
    def _monte_carlo_simulation(self, hole_card, community_cards, simulations=1500):
        try:
            hole_cards = [Card.from_str(card) for card in hole_card]
            community = [Card.from_str(card) for card in community_cards]
            
            all_cards = [Card.from_id(i) for i in range(1, 53)]
            used_card_ids = set()
            for card in hole_cards + community:
                used_card_ids.add(card.to_id())
            
            available_cards = [card for card in all_cards if card.to_id() not in used_card_ids]
            
            wins = 0
            ties = 0
            total_sims = 0
            
            for _ in range(simulations):
                try:
                    deck = available_cards.copy()
                    random.shuffle(deck)
                    
                    opp_hole = deck[:2]
                    remaining_deck = deck[2:]
                    
                    sim_community = community.copy()
                    cards_needed = 5 - len(sim_community)
                    if cards_needed > 0:
                        sim_community.extend(remaining_deck[:cards_needed])
                    
                    my_strength = HandEvaluator.eval_hand(hole_cards, sim_community)
                    opp_strength = HandEvaluator.eval_hand(opp_hole, sim_community)
                    
                    if my_strength > opp_strength:
                        wins += 1
                    elif my_strength == opp_strength:
                        ties += 1
                    
                    total_sims += 1
                    
                except Exception:
                    continue
            
            if total_sims == 0:
                return 0.5
            
            return (wins + ties * 0.5) / total_sims
            
        except Exception:
            return 0.5
    
    def _calculate_fold_ev(self, current_profit, remaining_rounds):
        return self._calculate_game_win_probability(current_profit, remaining_rounds - 1)
    
    def _calculate_call_ev(self, current_profit, remaining_rounds, call_amount, pot_size, win_prob, position):
        try:
            total_pot_after_call = pot_size + call_amount
            pot_odds = total_pot_after_call / call_amount if call_amount > 0 else 0
            required_equity = 1 / (1 + pot_odds) if pot_odds > 0 else 0.5
            
            win_profit = current_profit + total_pot_after_call - call_amount
            lose_profit = current_profit - call_amount
            
            win_game_prob = self._calculate_game_win_probability(win_profit, remaining_rounds - 1)
            lose_game_prob = self._calculate_game_win_probability(lose_profit, remaining_rounds - 1)
            
            base_ev = win_prob * win_game_prob + (1 - win_prob) * lose_game_prob
            
            if win_prob > required_equity * 0.95:
                pot_odds_bonus = 0.05
            else:
                pot_odds_bonus = 0
            
            position_bonus = 0.02 if position == "button" else 0
            
            return base_ev + pot_odds_bonus + position_bonus
            
        except Exception:
            return 0.0
    
    def _calculate_raise_ev(self, current_profit, remaining_rounds, valid_actions, pot_size, win_prob, position):
        try:
            min_raise = valid_actions[2]["amount"]["min"]
            max_raise = valid_actions[2]["amount"]["max"]
            
            if min_raise == -1:
                return 0.0, 0
            
            current_stack = current_profit + 1000
            
            best_ev = 0.0
            best_raise = min_raise
            
            if win_prob >= 0.75:
                test_sizes = [min_raise, pot_size * 0.75, pot_size * 1.0, current_stack]
            elif win_prob >= 0.60:
                test_sizes = [min_raise, pot_size * 0.5, pot_size * 0.75]
            else:
                test_sizes = [min_raise, pot_size * 0.3, pot_size * 0.5]
            
            for raise_size in test_sizes:
                raise_amount = max(min_raise, min(raise_size, max_raise))
                if raise_amount < min_raise:
                    continue
                
                fold_equity = self._estimate_fold_equity(raise_amount, pot_size, win_prob)
                
                fold_profit = current_profit + pot_size
                fold_ev = self._calculate_game_win_probability(fold_profit, remaining_rounds - 1)
                
                total_pot_if_called = pot_size + raise_amount * 2
                win_profit = current_profit + total_pot_if_called - raise_amount
                lose_profit = current_profit - raise_amount
                
                win_game_prob = self._calculate_game_win_probability(win_profit, remaining_rounds - 1)
                lose_game_prob = self._calculate_game_win_probability(lose_profit, remaining_rounds - 1)
                
                call_ev = win_prob * win_game_prob + (1 - win_prob) * lose_game_prob
                
                total_ev = fold_equity * fold_ev + (1 - fold_equity) * call_ev
                
                if position == "button":
                    total_ev += 0.01
                
                if total_ev > best_ev:
                    best_ev = total_ev
                    best_raise = raise_amount
            
            return best_ev, best_raise
            
        except Exception:
            return 0.0, min_raise if min_raise != -1 else 0
    
    def _estimate_fold_equity(self, raise_amount, pot_size, win_prob):
        try:
            if pot_size == 0:
                bet_to_pot_ratio = 2.0
            else:
                bet_to_pot_ratio = raise_amount / pot_size
            
            if bet_to_pot_ratio <= 0.5:
                base_fold_rate = 0.2
            elif bet_to_pot_ratio <= 1.0:
                base_fold_rate = 0.35
            elif bet_to_pot_ratio <= 2.0:
                base_fold_rate = 0.5
            else:
                base_fold_rate = 0.65
            
            if win_prob < 0.3:
                base_fold_rate *= 0.8
            elif win_prob > 0.8:
                base_fold_rate *= 1.2
            
            opponent_fold_tendency = self._get_opponent_fold_tendency()
            adjusted_fold_rate = base_fold_rate * opponent_fold_tendency
            
            return max(0.05, min(0.8, adjusted_fold_rate))
            
        except Exception:
            return 0.3
    
    def _get_opponent_fold_tendency(self):
        try:
            if self.total_opponent_actions < 10:
                return 1.0
            
            fold_rate = self.opponent_folds / self.total_opponent_actions
            
            if fold_rate > 0.6:
                return 1.3
            elif fold_rate < 0.3:
                return 0.7
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_game_win_probability(self, current_profit, remaining_rounds):
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
            if z < -6:
                return 0.0
            if z > 6:
                return 1.0
            
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911
            
            sign = 1 if z >= 0 else -1
            z = abs(z)
            
            t = 1.0 / (1.0 + p * z)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
            
            return 0.5 * (1.0 + sign * y)
            
        except Exception:
            return 0.5
    
    def _get_position(self, round_state):
        try:
            seats = round_state.get('seats', [])
            dealer_btn = round_state.get('dealer_btn', 0)
            
            for i, seat in enumerate(seats):
                if seat.get('uuid') == self.uuid:
                    if i == dealer_btn:
                        return "button"
                    elif i == (dealer_btn + 1) % len(seats):
                        return "small_blind"
                    elif i == (dealer_btn + 2) % len(seats):
                        return "big_blind"
                    else:
                        return "early"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def _get_pot_size(self, round_state):
        try:
            pot_info = round_state.get('pot', {})
            main_pot = pot_info.get('main', {}).get('amount', 0)
            side_pots = pot_info.get('side', [])
            
            total_side = sum(side.get('amount', 0) for side in side_pots)
            return main_pot + total_side
            
        except Exception:
            return 0
    
    def _get_my_stack(self, round_state):
        try:
            for seat in round_state.get('seats', []):
                if seat.get('uuid') == self.uuid:
                    return seat.get('stack', 1000)
            return 1000
        except Exception:
            return 1000
    
    def _get_remaining_rounds(self, round_state):
        try:
            current_round = round_state.get('round_count', 1)
            return max(1, 20 - current_round + 1)
        except Exception:
            return 10
    
    def _parse_rank(self, rank_char):
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(rank_char, 2)
    
    def _track_opponent_action(self, action):
        try:
            self.total_opponent_actions += 1
            
            action_type = action.get('action', '')
            if action_type == 'RAISE':
                self.opponent_raises += 1
            elif action_type == 'CALL':
                self.opponent_calls += 1
            elif action_type == 'FOLD':
                self.opponent_folds += 1
                
        except Exception:
            pass
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        try:
            if action.get('player_uuid') != self.uuid:
                self._track_opponent_action(action)
        except Exception:
            pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            won = any(winner.get('uuid') == self.uuid for winner in winners)
            
            self.recent_performance.append(1 if won else 0)
            if len(self.recent_performance) > 5:
                self.recent_performance.pop(0)
            
            if len(self.recent_performance) == 5:
                win_rate = sum(self.recent_performance) / 5
                if win_rate < 0.3:
                    self.mu = min(5.0, self.mu * 1.02)
                elif win_rate > 0.7:
                    self.mu = max(3.5, self.mu * 0.99)
                    
        except Exception:
            pass


def setup_ai():
    return PBV2PokerAgent()