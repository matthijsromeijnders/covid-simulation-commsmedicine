# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import sys
#import resource
from ClassT import ModelT
import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz
# ----------------------------------------------------------------- #
# Initialize Class
# ----------------------------------------------------------------- #

# Interventions:
    # 'ref'
    # 'working'
    # 'behavior'

    # 'school' 
    # 'schoolisolation'
    # 'schoolparents'
    # 'schoolextreme'

    # 'G4'
    # 'border'
    # 'local'
    # 'brablim'
    
    # 'self-isolation'

# some parameters

# Current idea is primary school, students, working. [1] [3] [5]

initial_loc = int(sys.argv[1])

runs_done = int(sys.argv[2])
runs = int(sys.argv[3])

intervention = str(sys.argv[4])
adherence = float(sys.argv[5])
print('this sim does: muni: ', initial_loc, ' runs: ', runs, ' intervention: ', intervention, ' adherence: ', adherence)

# For WEIBULL RUNS latent = 2.22, infect = 5.64
latent = 0.5
incub = 100
infect = 3.2
gamma = True
n_initial_infect = 5
self_isolation_period = 3 # set to self isolation period in days, (its the period it takes after infection when people start to self isolate)
city_size = 100000
# If doing transmission runs, set no_group to True.
print(0.135 * 0.405)

for interv in [intervention]:
	seed_list = []
	for seed in [1,2,3,4,5]:
		for demo_group in [5]:
			demo_run_list = []
			for run in range(runs_done, runs):
				params_input = {'savename': interv,
                                	'intervention': interv,
                                	'Ndays': 21 * 24,
                                	'seed': seed,
                                	'self_isolation_period': self_isolation_period,
					'big_city_threshold': city_size,
					'respect_border_lockdown_probability': adherence,
					'gamma': gamma
					}
				ClassT = ModelT(params_input)
				print(ClassT.big_city_threshold)
				ClassT.read_model_data(initial_loc, demo_group, n_initial_infect, no_group=True)
				ClassT.read_empirical_data()
				ClassT.set_parameters(latent, incub, infect)
				ClassT.initialise(demo_group)
				ClassT.simulate_new()
				status_file = ClassT.save(run, demo_group, initial_loc)
				del ClassT
				demo_run_list.append(status_file)
			array_tot = np.zeros((355,504))
			for i in range(len(demo_run_list)):
				array_tot += demo_run_list[i]
			dense_average = array_tot / len(demo_run_list)
			dense_std = np.std(dense_average, axis=0)
			print('PRINT STATUS SHAPE ::::   ',dense_average.shape)
			# Convert the dense average back to sparse format if n
			path = f'../Data/Model_V1/Data/{interv}/Seed_{seed}/Runs_{interv}_{latent}_{infect}_risk{initial_loc}_{demo_group}/'
			if not os.path.exists(path):
				os.makedirs(path)
			np.save(path + f'Status_avg_runs={runs}_{city_size}_{adherence}', dense_average)
			np.save(path + f'Status_std_runs={runs}_{city_size}_{adherence}', dense_std)
	dense_arrays = np.zeros((355,504))
	dense_std = np.std(dense_arrays, axis=0)
	for i in range(len(seed_list)):
		dense_arrays += seed_list[i]
	dense_arrays /= len(seed_list)
	path = f'../Data/Model_V1/Data/{interv}/Seed_0/muni={initial_loc}'	
	np.save(path + f'Status_avg_runs=total_{city_size}_{adherence}', dense_average)











