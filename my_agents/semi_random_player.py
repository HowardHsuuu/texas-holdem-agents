from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
import random
import math
import time
from collections import deque

class ChaosPokerAI(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        print("Initializing Chaos Poker AI...")
        
        self.chaos_level = 0.7
        self.mood_swings = True
        self.current_mood = "neutral"
        self.mood_duration = 0
        
        self.decision_patterns = [
            "aggressive_spree", "tight_phase", "bluff_heavy", 
            "call_station", "fold_happy", "balanced"
        ]
        self.current_pattern = random.choice(self.decision_patterns)
        self.pattern_rounds_left = random.randint(3, 8)
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.position = None
        self.round_count = 0
        self.games_played = 0
        
        self.recent_actions = deque(maxlen=10)
        self.superstitions = self._generate_superstitions()
        self.lucky_numbers = [random.randint(1, 20) for _ in range(3)]
        
        self.wins = 0
        self.losses = 0
        self.biggest_bluff = 0
        self.craziest_call = 0
        
        print("Chaos AI ready to create mayhem!")
        self._print_chaos_status()
    
    def _generate_superstitions(self):
        superstitions = []
        possible_superstitions = [
            ("red_cards_lucky", "Red cards bring good luck"),
            ("suited_unlucky", "Suited hands are cursed"),
            ("sevens_magic", "Sevens are magical"),
            ("even_numbers_bad", "Even ranks are unlucky"),
            ("flop_hearts_fold", "If flop has hearts, must fold"),
            ("big_pot_bluff", "Big pots demand bluffs"),
            ("position_matters", "Position determines destiny"),
            ("time_based", "Time of decision affects luck")
        ]
        
        num_superstitions = random.randint(2, 3)
        selected = random.sample(possible_superstitions, num_superstitions)
        
        for sup_type, description in selected:
            superstitions.append({
                'type': sup_type,
                'description': description,
                'active': random.choice([True, False])
            })
        
        return superstitions
    
    def _print_chaos_status(self):
        print("Chaos AI Configuration:")
        print(f"   Chaos level: {self.chaos_level:.1f}")
        print(f"   Current mood: {self.current_mood}")
        print(f"   Pattern: {self.current_pattern} ({self.pattern_rounds_left} rounds left)")
        print(f"   Lucky numbers: {self.lucky_numbers}")
        print("   Active superstitions:")
        for sup in self.superstitions:
            if sup['active']:
                print(f"     - {sup['description']}")
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            
            self._update_mood_and_patterns()
            
            hand_strength = self._evaluate_hand_with_chaos(hole_card, round_state)
            pot_size = self._get_pot_size(round_state)
            call_amount = valid_actions[1]['amount']
            
            superstition_modifier = self._check_superstitions(hole_card, round_state)
            
            base_decision = self._pattern_based_decision(
                valid_actions, hand_strength, pot_size, call_amount, round_state
            )
            
            final_decision = self._apply_chaos_modifier(
                base_decision, valid_actions, superstition_modifier
            )
            
            self._track_decision(final_decision, hand_strength, pot_size)
            
            print(f"Chaos: {self.current_mood} {self.current_pattern} | "
                  f"Hand {hand_strength:.2f} | Superstition {superstition_modifier:+.2f} | "
                  f"Action: {final_decision[0]} {final_decision[1]}")
            
            return final_decision
            
        except Exception as e:
            print(f"Chaos AI error (even chaos has limits): {e}")
            return self._emergency_chaos_action(valid_actions)
    
    def _evaluate_hand_with_chaos(self, hole_card, round_state):
        if not hole_card or len(hole_card) != 2:
            return random.uniform(0.2, 0.8)
        
        try:
            from game.engine.card import Card
            hole_cards = [Card.from_str(card) for card in hole_card]
            
            community_str = round_state.get('community_card', [])
            community_cards = [Card.from_str(card) for card in community_str]
            
            if len(hole_cards) == 2:
                hand_score = HandEvaluator.eval_hand(hole_cards, community_cards)
                
                normalized_strength = min(0.95, max(0.1, (hand_score / 100000000.0)))
                
                if normalized_strength < 0.1 or normalized_strength > 0.95:
                    normalized_strength = self._basic_hand_strength(hole_card)
            else:
                normalized_strength = self._basic_hand_strength(hole_card)
                
        except Exception as e:
            print(f"Hand evaluation chaos: {e}")
            normalized_strength = self._basic_hand_strength(hole_card)
        
        chaos_modifier = random.uniform(-0.3, 0.3) * self.chaos_level
        
        mood_modifiers = {
            "euphoric": 0.2,
            "confident": 0.1,
            "neutral": 0.0,
            "pessimistic": -0.1,
            "paranoid": -0.2
        }
        mood_modifier = mood_modifiers.get(self.current_mood, 0.0)
        
        ranks = [self._get_rank(card) for card in hole_card]
        luck_modifier = 0.0
        for rank in ranks:
            if rank in self.lucky_numbers:
                luck_modifier += 0.15
                print(f"Lucky number {rank} detected!")
        
        final_strength = normalized_strength + chaos_modifier + mood_modifier + luck_modifier
        return max(0.1, min(0.95, final_strength))
    
    def _basic_hand_strength(self, hole_card):
        ranks = [self._get_rank(card) for card in hole_card]
        suits = [card[0] for card in hole_card]
        
        if ranks[0] == ranks[1]:
            pair_strength = {
                14: 0.95, 13: 0.90, 12: 0.85, 11: 0.80, 10: 0.75,
                9: 0.65, 8: 0.55, 7: 0.45, 6: 0.35, 5: 0.25,
                4: 0.20, 3: 0.15, 2: 0.10
            }
            base = pair_strength.get(ranks[0], 0.3)
        else:
            high_rank = max(ranks)
            low_rank = min(ranks)
            
            if high_rank == 14:
                ace_strength = {13: 0.85, 12: 0.80, 11: 0.75, 10: 0.70}
                base = ace_strength.get(low_rank, 0.4 + low_rank * 0.02)
            elif high_rank >= 12:
                base = 0.3 + (high_rank + low_rank) * 0.02
            else:
                base = 0.2 + (high_rank + low_rank) * 0.015
        
        if suits[0] == suits[1]:
            suited_bonus = random.uniform(0.05, 0.15)
            base += suited_bonus
        
        if abs(ranks[0] - ranks[1]) <= 3:
            connector_bonus = random.uniform(0.02, 0.08)
            base += connector_bonus
        
        return base
    
    def _update_mood_and_patterns(self):
        if self.mood_swings:
            self.mood_duration -= 1
            if self.mood_duration <= 0:
                moods = ["euphoric", "confident", "neutral", "pessimistic", "paranoid"]
                old_mood = self.current_mood
                self.current_mood = random.choice(moods)
                self.mood_duration = random.randint(2, 6)
                
                if old_mood != self.current_mood:
                    print(f"Mood swing: {old_mood} → {self.current_mood}")
        
        self.pattern_rounds_left -= 1
        if self.pattern_rounds_left <= 0:
            old_pattern = self.current_pattern
            self.current_pattern = random.choice(self.decision_patterns)
            self.pattern_rounds_left = random.randint(3, 8)
            
            if old_pattern != self.current_pattern:
                print(f"Pattern shift: {old_pattern} → {self.current_pattern}")
    
    def _check_superstitions(self, hole_card, round_state):
        modifier = 0.0
        
        for superstition in self.superstitions:
            if not superstition['active']:
                continue
            
            sup_type = superstition['type']
            
            if sup_type == "red_cards_lucky":
                red_cards = sum(1 for card in hole_card if card[0] in ['H', 'D'])
                if red_cards == 2:
                    modifier += 0.2
                    print("All red cards - feeling lucky!")
            
            elif sup_type == "suited_unlucky":
                if hole_card[0][0] == hole_card[1][0]:
                    modifier -= 0.15
                    print("Suited cards are cursed!")
            
            elif sup_type == "sevens_magic":
                sevens = sum(1 for card in hole_card if card[1] == '7')
                modifier += sevens * 0.1
                if sevens > 0:
                    print(f"{sevens} magical seven(s)!")
            
            elif sup_type == "even_numbers_bad":
                even_ranks = sum(1 for card in hole_card 
                               if self._get_rank(card) % 2 == 0)
                modifier -= even_ranks * 0.08
                if even_ranks > 0:
                    print(f"{even_ranks} unlucky even number(s)")
            
            elif sup_type == "flop_hearts_fold":
                community = round_state.get('community_card', [])
                if len(community) >= 3:
                    hearts_on_flop = sum(1 for card in community[:3] if card[0] == 'H')
                    if hearts_on_flop >= 2:
                        modifier -= 0.5
                        print("Too many hearts on flop - bad omen!")
            
            elif sup_type == "big_pot_bluff":
                pot_size = self._get_pot_size(round_state)
                if pot_size > 100:
                    modifier += 0.3
                    print("Big pot demands a bluff!")
            
            elif sup_type == "time_based":
                current_time = int(time.time()) % 10
                if current_time in [3, 7]:
                    modifier += 0.15
                    print("The stars align favorably!")
                elif current_time in [1, 9]:
                    modifier -= 0.1
                    print("Bad timing...")
        
        return modifier
    
    def _pattern_based_decision(self, valid_actions, hand_strength, pot_size, call_amount, round_state):
        pattern = self.current_pattern
        
        if pattern == "aggressive_spree":
            return self._aggressive_decision(valid_actions, hand_strength, pot_size)
        
        elif pattern == "tight_phase":
            return self._tight_decision(valid_actions, hand_strength, call_amount)
        
        elif pattern == "bluff_heavy":
            return self._bluff_heavy_decision(valid_actions, hand_strength, pot_size)
        
        elif pattern == "call_station":
            return self._call_station_decision(valid_actions, call_amount)
        
        elif pattern == "fold_happy":
            return self._fold_happy_decision(valid_actions, hand_strength, call_amount)
        
        else:
            return self._balanced_decision(valid_actions, hand_strength, pot_size, call_amount)
    
    def _aggressive_decision(self, valid_actions, hand_strength, pot_size):
        if hand_strength > 0.3:
            return self._make_raise(valid_actions, pot_size, random.uniform(0.6, 1.2))
        elif hand_strength > 0.15:
            return 'call', valid_actions[1]['amount']
        else:
            return 'fold', 0
    
    def _tight_decision(self, valid_actions, hand_strength, call_amount):
        if hand_strength > 0.8:
            return self._make_raise(valid_actions, call_amount * 2, random.uniform(0.3, 0.6))
        elif hand_strength > 0.6 and call_amount <= 20:
            return 'call', call_amount
        else:
            return 'fold', 0
    
    def _bluff_heavy_decision(self, valid_actions, hand_strength, pot_size):
        if random.random() < 0.6:
            return self._make_raise(valid_actions, pot_size, random.uniform(0.4, 0.8))
        elif hand_strength > 0.4:
            return 'call', valid_actions[1]['amount']
        else:
            return 'fold', 0
    
    def _call_station_decision(self, valid_actions, call_amount):
        if call_amount <= self.my_stack * 0.2:
            return 'call', call_amount
        else:
            return 'fold', 0
    
    def _fold_happy_decision(self, valid_actions, hand_strength, call_amount):
        if hand_strength > 0.85 and call_amount <= 50:
            return 'call', call_amount
        else:
            return 'fold', 0
    
    def _balanced_decision(self, valid_actions, hand_strength, pot_size, call_amount):
        if hand_strength > 0.7:
            return self._make_raise(valid_actions, pot_size, random.uniform(0.4, 0.7))
        elif hand_strength > 0.4 and call_amount <= pot_size * 0.5:
            return 'call', call_amount
        else:
            return 'fold', 0
    
    def _apply_chaos_modifier(self, base_decision, valid_actions, superstition_modifier):
        action, amount = base_decision
        
        chaos_chance = self.chaos_level * 0.3 + abs(superstition_modifier) * 0.2
        
        if random.random() < chaos_chance:
            print("CHAOS OVERRIDE!")
            
            if random.random() < 0.4:
                new_actions = ['fold', 'call', 'raise']
                available_actions = [a['action'] for a in valid_actions]
                possible_actions = [a for a in new_actions if a in available_actions]
                
                if possible_actions:
                    action = random.choice(possible_actions)
                    
                    if action == 'fold':
                        amount = 0
                    elif action == 'call':
                        amount = valid_actions[1]['amount']
                    elif action == 'raise':
                        raise_info = valid_actions[2]['amount']
                        if raise_info['min'] != -1:
                            min_raise = raise_info['min']
                            max_raise = raise_info['max']
                            chaos_multiplier = random.uniform(0.1, 2.0)
                            pot_size = self._get_pot_size({'pot': {'main': {'amount': 50}, 'side': []}})
                            target = min_raise + int(pot_size * chaos_multiplier)
                            amount = max(min_raise, min(target, max_raise))
                        else:
                            action = 'call'
                            amount = valid_actions[1]['amount']
            
            if action == 'raise' and random.random() < 0.3:
                raise_info = valid_actions[2]['amount']
                if raise_info['min'] != -1:
                    modifier = random.uniform(0.5, 1.8)
                    amount = int(amount * modifier)
                    amount = max(raise_info['min'], min(amount, raise_info['max']))
        
        return action, amount
    
    def _make_raise(self, valid_actions, reference_amount, multiplier):
        if len(valid_actions) < 3:
            return 'call', valid_actions[1]['amount']
        
        raise_info = valid_actions[2]['amount']
        if raise_info['min'] == -1:
            return 'call', valid_actions[1]['amount']
        
        min_raise = raise_info['min']
        max_raise = raise_info['max']
        
        target_raise = min_raise + int(reference_amount * multiplier)
        final_raise = max(min_raise, min(target_raise, max_raise))
        
        return 'raise', int(final_raise)
    
    def _track_decision(self, decision, hand_strength, pot_size):
        action, amount = decision
        self.recent_actions.append({
            'action': action,
            'amount': amount,
            'hand_strength': hand_strength,
            'pot_size': pot_size
        })
        
        if action == 'raise' and hand_strength < 0.3:
            self.biggest_bluff = max(self.biggest_bluff, amount)
        elif action == 'call' and amount > pot_size:
            self.craziest_call = max(self.craziest_call, amount)
    
    def _emergency_chaos_action(self, valid_actions):
        if random.random() < 0.7:
            return 'call', valid_actions[1]['amount']
        else:
            return 'fold', 0
    
    def _get_rank(self, card_str):
        if len(card_str) < 2:
            return 2
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(card_str[1], 2)
    
    def _get_pot_size(self, round_state):
        pot_info = round_state.get('pot', {})
        main_pot = pot_info.get('main', {}).get('amount', 0)
        side_pots = sum(side.get('amount', 0) for side in pot_info.get('side', []))
        return main_pot + side_pots
    
    def _update_game_state(self, round_state):
        seats = round_state.get('seats', [])
        for seat in seats:
            if seat.get('uuid') == self.uuid:
                self.my_stack = seat.get('stack', self.my_stack)
                sb_pos = round_state.get('small_blind_pos', 0)
                if seats.index(seat) == sb_pos:
                    self.position = 'SB'
                else:
                    self.position = 'BB'
            else:
                self.opponent_stack = seat.get('stack', self.opponent_stack)
    
    def _print_chaos_stats(self):
        print("Chaos Stats:")
        print(f"   Games: {self.wins}W-{self.losses}L")
        print(f"   Biggest bluff: {self.biggest_bluff}")
        print(f"   Craziest call: {self.craziest_call}")
        print(f"   Current pattern: {self.current_pattern}")
        print(f"   Mood: {self.current_mood}")
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        self.round_count = 0
        self.my_stack = 1000
        self.opponent_stack = 1000
        
        if random.random() < 0.3:
            print("Adopting new superstitions...")
            self.superstitions = self._generate_superstitions()
        
        print(f"Chaos AI entering game #{self.games_played}")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        
        for seat in seats:
            if seat.get('uuid') == self.uuid:
                self.my_stack = seat.get('stack', self.my_stack)
            else:
                self.opponent_stack = seat.get('stack', self.opponent_stack)
    
    def receive_street_start_message(self, street, round_state):
        comments = {
            'flop': ["The chaos begins!", "Let's roll the dice!", "Flop tornado!"],
            'turn': ["Plot twist!", "Turn time!", "Getting spicy!"],
            'river': ["River of destiny!", "Final act!", "Showdown approaching!"]
        }
        
        if street in comments:
            print(random.choice(comments[street]))
    
    def receive_game_update_message(self, action, round_state):
        player_uuid = action.get('player_uuid')
        if player_uuid != self.uuid:
            action_type = action.get('action', '').lower()
            
            reactions = {
                'fold': ["Coward!", "Victory!", "Did I scare them?"],
                'call': ["Interesting...", "Playing safe?", "Taking the bait!"],
                'raise': ["Challenge accepted!", "War!", "To the moon!"]
            }
            
            if action_type in reactions:
                print(random.choice(reactions[action_type]))
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        won = any(winner.get('uuid') == self.uuid for winner in winners)
        
        if won:
            self.wins += 1
            reactions = ["CHAOS REIGNS!", "Tornado victory!", "Lucky chaos!", 
                        "Muahahaha!", "Method to my madness!"]
            print(random.choice(reactions))
        else:
            self.losses += 1
            reactions = ["Temporary setback!", "Chaos finds a way!", "Unlucky roll!",
                        "This isn't over!", "The stars were misaligned!"]
            print(random.choice(reactions))
        
        if self.round_count % 15 == 0:
            self._print_chaos_stats()

def setup_ai():
    return ChaosPokerAI()