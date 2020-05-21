# -------- import GridSearch and define/import the compile function -------- #
import sys
sys.path.append('SIGS-Grid-Search')
from grid_search import GridSearch

# -------- main file to run -------- #
main_file = '-m spinup.run'

# -------- define dictionary of arguments for grid search -------- #
args = {'algo': ['td3', 'ddpg'],
	'env': ['Hopper-v3', 'Ant-v2', 'Humanoid-v2'],
	'steps_per_epoch': [2000],
	'epochs': [1000]}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=6)
myGridSearch.run()
