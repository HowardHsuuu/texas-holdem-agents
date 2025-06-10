from game.players import BasePokerPlayer
import random

class AllInMonsterAI(BasePokerPlayer):
    
    def __init__(self):
        super().__init__()
        print("Initializing ALL-IN MONSTER...")
        print("WARNING: This AI goes ALL-IN every single hand!")
        
        self.all_ins = 0
        self.folds_forced = 0
        self.showdowns_won = 0
        self.showdowns_lost = 0
        self.total_chips_won = 0
        self.biggest_bluff_win = 0
        
        self.battle_cries = [
            "ALL-IN OR NOTHING!",
            "TO THE MOON!",
            "MAXIMUM PRESSURE!",
            "FEAR THE MONSTER!",
            "DIAMOND HANDS!",
            "NO MERCY!",
            "BATTLE MODE!",
            "CHAOS INCARNATE!"
        ]
        
        self.victory_taunts = [
            "BOW TO THE MONSTER!",
            "EASY MONEY!",
            "FEAR TACTICS WORK!",
            "ANOTHER VICTIM!",
            "UNSTOPPABLE FORCE!"
        ]
        
        self.defeat_responses = [
            "SOMEONE HAD THE GUTS!",
            "FINALLY, A WORTHY OPPONENT!",
            "VARIANCE HAPPENS!",
            "HONOR TO THE BRAVE!",
            "I'LL BE BACK!"
        ]
        
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.round_count = 0
        self.games_played = 0
        self.previous_opponent_stack = 1000
        
        print("ALL-IN MONSTER ACTIVATED!")
        print("Opponents will learn to fear the all-in...")
        self._print_monster_stats()
    
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._update_game_state(round_state)
            
            battle_cry = random.choice(self.battle_cries)
            print(f"MONSTER: {battle_cry}")
            
            if len(valid_actions) >= 3:
                raise_info = valid_actions[2]['amount']
                
                if isinstance(raise_info, dict) and raise_info.get('min', -1) != -1:
                    min_raise = raise_info['min']
                    max_raise = raise_info['max']
                    
                    all_in_amount = max_raise
                    
                    all_in_amount = min(all_in_amount, self.my_stack + self._get_current_bet(round_state))
                    all_in_amount = max(min_raise, all_in_amount)
                    
                    self.all_ins += 1
                    
                    card_str = f"[{hole_card[0]}, {hole_card[1]}]" if hole_card and len(hole_card) == 2 else "[??]"
                    print(f"ALL-IN WITH {card_str} - AMOUNT: {all_in_amount}")
                    print(f"THIS IS ALL-IN #{self.all_ins}!")
                    
                    return 'raise', all_in_amount
                else:
                    print("CAN'T RAISE - FALLING BACK TO CALL!")
            
            call_amount = valid_actions[1]['amount']
            
            if call_amount >= self.my_stack * 0.8:
                print(f"CALLING ALL-IN: {call_amount}")
                print("SOMEONE BEAT ME TO THE PUNCH!")
            else:
                print(f"FORCED TO JUST CALL: {call_amount}")
                print("THIS ISN'T MY FULL POWER!")
            
            return 'call', call_amount
            
        except Exception as e:
            print(f"MONSTER ERROR: {e}")
            return self._emergency_monster_action(valid_actions)
    
    def _get_current_bet(self, round_state):
        try:
            action_histories = round_state.get('action_histories', {})
            current_street = round_state.get('street', 'preflop')
            
            street_actions = action_histories.get(current_street, [])
            
            my_current_bet = 0
            for action in street_actions:
                if action.get('uuid') == self.uuid and 'amount' in action:
                    my_current_bet = action.get('amount', 0)
            
            return my_current_bet
            
        except Exception as e:
            print(f"Error getting current bet: {e}")
            return 0
    
    def _emergency_monster_action(self, valid_actions):
        print("MONSTER IN EMERGENCY MODE!")
        
        try:
            if len(valid_actions) >= 3:
                raise_info = valid_actions[2]['amount']
                if isinstance(raise_info, dict) and raise_info.get('min', -1) != -1:
                    max_raise = raise_info['max']
                    print(f"EMERGENCY ALL-IN: {max_raise}")
                    return 'raise', max_raise
            
            call_amount = valid_actions[1]['amount']
            print(f"EMERGENCY CALL: {call_amount}")
            return 'call', call_amount
            
        except Exception as e:
            print(f"EMERGENCY ERROR: {e}")
            print("MONSTER FORCED TO FOLD!")
            return 'fold', 0
    
    def _update_game_state(self, round_state):
        try:
            old_opponent_stack = self.opponent_stack
            
            seats = round_state.get('seats', [])
            for seat in seats:
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
            
            if (old_opponent_stack > self.opponent_stack and 
                self.all_ins > 0 and 
                old_opponent_stack - self.opponent_stack > 5):
                
                chips_gained = old_opponent_stack - self.opponent_stack
                self.folds_forced += 1
                self.total_chips_won += chips_gained
                
                if chips_gained > self.biggest_bluff_win:
                    self.biggest_bluff_win = chips_gained
                
                print(f"FORCED FOLD! Won {chips_gained} chips!")
                print(random.choice(self.victory_taunts))
                
        except Exception as e:
            print(f"Error updating game state: {e}")
    
    def _print_monster_stats(self):
        print("ALL-IN MONSTER STATISTICS:")
        print(f"Total all-ins: {self.all_ins}")
        print(f"Folds forced: {self.folds_forced}")
        print(f"Showdowns won: {self.showdowns_won}")
        print(f"Showdowns lost: {self.showdowns_lost}")
        print(f"Total chips terrorized: {self.total_chips_won}")
        print(f"Biggest bluff win: {self.biggest_bluff_win}")
        
        if self.all_ins > 0:
            fold_rate = self.folds_forced / self.all_ins * 100
            print(f"Opponent fold rate: {fold_rate:.1f}%")
            
            total_showdowns = self.showdowns_won + self.showdowns_lost
            if total_showdowns > 0:
                showdown_win_rate = self.showdowns_won / total_showdowns * 100
                print(f"Showdown win rate: {showdown_win_rate:.1f}%")
        
        print("FEAR THE MONSTER!")
    
    def _analyze_opponent_courage(self):
        total_showdowns = self.showdowns_won + self.showdowns_lost
        
        if total_showdowns == 0:
            courage_level = "TERRIFIED (never called)"
        elif self.folds_forced == 0:
            courage_level = "FEARLESS (always calls)"
        elif self.all_ins > 0:
            fold_percentage = self.folds_forced / self.all_ins
            if fold_percentage > 0.8:
                courage_level = "CHICKEN (folds 80%+)"
            elif fold_percentage > 0.5:
                courage_level = "CAUTIOUS (folds 50%+)"
            else:
                courage_level = "BRAVE (calls most all-ins)"
        else:
            courage_level = "UNKNOWN (no data)"
        
        print(f"OPPONENT COURAGE LEVEL: {courage_level}")
        return courage_level
    
    def receive_game_start_message(self, game_info):
        self.games_played += 1
        self.round_count = 0
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.previous_opponent_stack = 1000
        
        print(f"ALL-IN MONSTER ENTERING GAME #{self.games_played}")
        print("PREPARE FOR MAXIMUM AGGRESSION!")
        
        print(f"Career stats: {self.all_ins} all-ins, {self.folds_forced} folds forced")
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        
        try:
            for seat in seats:
                if seat.get('uuid') == self.uuid:
                    self.my_stack = seat.get('stack', self.my_stack)
                else:
                    self.opponent_stack = seat.get('stack', self.opponent_stack)
        except Exception as e:
            print(f"Error updating stacks: {e}")
        
        if hole_card and len(hole_card) == 2:
            print(f"Round {round_count}: Monster holds {hole_card[0]}, {hole_card[1]}")
            print("Doesn't matter what cards I have - ALL-IN INCOMING!")
        else:
            print(f"Round {round_count}: Cards are irrelevant - ALL-IN INCOMING!")
    
    def receive_street_start_message(self, street, round_state):
        street_taunts = {
            'flop': [
                "FLOP? MORE LIKE FLOP SWEAT!",
                "Three cards can't save you!",
                "STILL GOING ALL-IN!"
            ],
            'turn': [
                "TURN UP THE PRESSURE!",
                "One more card, same strategy!",
                "MONSTER DOESN'T CHANGE PLANS!"
            ],
            'river': [
                "RIVER OF TEARS!",
                "FINAL CHANCE TO FOLD!",
                "SHOWDOWN WITH THE MONSTER!"
            ]
        }
        
        if street in street_taunts:
            print(random.choice(street_taunts[street]))
    
    def receive_game_update_message(self, action, round_state):
        try:
            player_uuid = action.get('player_uuid')
            if player_uuid != self.uuid:
                action_type = action.get('action', '').lower()
                amount = action.get('amount', 0)
                
                if action_type == 'fold':
                    print("ANOTHER VICTIM FALLS!")
                    print("THE MONSTER'S REPUTATION GROWS!")
                    
                elif action_type == 'call':
                    if amount >= self.my_stack * 0.8:
                        print("SOMEONE HAS THE GUTS!")
                        print("FINALLY, A WORTHY BATTLE!")
                        print("MAY THE BEST HAND WIN!")
                    else:
                        print("Small call... STILL DANGEROUS!")
                        
                elif action_type == 'raise':
                    print("COUNTER-ATTACK DETECTED!")
                    print("THE MONSTER RESPECTS YOUR COURAGE!")
                    print("BUT I'LL STILL GO ALL-IN NEXT!")
                    
        except Exception as e:
            print(f"Error processing opponent action: {e}")
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        try:
            won = any(winner.get('uuid') == self.uuid for winner in winners)
            
            was_showdown = len(hand_info) > 0
            
            if won:
                if was_showdown:
                    self.showdowns_won += 1
                    print("MONSTER WINS AT SHOWDOWN!")
                    print(random.choice(self.victory_taunts))
                    print("FEAR + GOOD CARDS = UNBEATABLE!")
                else:
                    print("MONSTER WINS BY INTIMIDATION!")
                    print("OPPONENT CHICKENED OUT!")
            else:
                if was_showdown:
                    self.showdowns_lost += 1
                    print(random.choice(self.defeat_responses))
                    print("Cards didn't cooperate this time!")
                    print("BUT THE MONSTER WILL RETURN!")
                else:
                    print("Lost without showdown - probably a technical issue!")
                    print("MONSTER IS CONFUSED BUT STILL HUNGRY!")
            
            if self.round_count % 10 == 0:
                print("\n" + "="*50)
                self._print_monster_stats()
                self._analyze_opponent_courage()
                print("="*50 + "\n")
                
        except Exception as e:
            print(f"Error processing round result: {e}")

def setup_ai():
    return AllInMonsterAI()