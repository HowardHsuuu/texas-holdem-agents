import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.rule_player import setup_ai as rule_ai
from agents.monte_rule_player import setup_ai as monte_rule_ai
from agents.nn_player import setup_ai as nn_ai
from agents.mcts_player import setup_ai as mcts_ai
#from baseline0 import setup_ai as baseline0_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="p1", algorithm=mcts_ai())
config.register_player(name="p2", algorithm=monte_rule_ai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)

print(json.dumps(game_result, indent=4))
