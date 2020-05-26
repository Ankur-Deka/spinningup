# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
main_file = '-m spinup.run'

# -------- define dictionary of arguments for grid search -------- #
args = {'algo': ['ppo', 'vpg', 'sac'],
	'env': ['Pong-ram-v0'],
	'steps_per_epoch': [4000],
	'epochs': [10],
	'seed': [0]}

# -------- create GridSearch object and run -------- #
import grid_search
print(grid_search)
myGridSearch = GridSearch(main_file, args, num_process=8)
myGridSearch.run()
