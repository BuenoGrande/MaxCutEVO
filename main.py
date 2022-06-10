import numpy as np
import os

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import multiprocessing

crossovers = ["CustomCrossoverPotential","CustomCrossoverCluster", "UniformCrossover", "OnePointCrossover", "TwoPointCrossover"]

sets=['setA','setB','setC','setD','setE']

instance_set_A=["6","12","25","50","100"]
instance_set_B=["9","16","25","49","100","196","400","784","1600"]
instance_set_C=["6","12","25","50","100","200"]
instance_set_DE=["10","20","40","80","160"]

def run_instance(inst, visualize=False, verbose=False):
	results_success = []
	results_true = []
	for cx in crossovers:
		with open("output-{}.txt".format(cx), "w") as f:
			population_size = 500
			num_evaluations_list = []
			num_runs = 30
			num_success = 0
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				if visualize:
					fitness.visualize()
				genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)

				best_fitness, num_evaluations = genetic_algorithm.run()
				if best_fitness == fitness.value_to_reach:
					num_success += 1
			num_evaluations_list.append(num_evaluations)
			results_success.append(num_success)
			results_true.append(num_runs)
			if verbose:
				print("{}/{} runs successful".format(num_success, num_runs))
				print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
			f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
											  percentiles[2]))
	return results_success, results_true


def run_set(set_file, instance_size, visualize, verbose):
	path = os.path.join("maxcut-instances", set_file+ "/"+ instance_size)

	correct_instances = np.array(np.zeros(len(crossovers)))
	num_instances = np.array(np.zeros(len(crossovers)))

	list_files = []
	for instance in os.listdir(path):
		if instance.endswith(".txt"):
			list_files.append(os.path.join(path, instance))

	# Multiprocessing the running of files
	pool = multiprocessing.Pool()
	processes = [pool.apply_async(run_instance_helper, args=(file, visualize, verbose)) for file in list_files]
	result = [p.get() for p in processes]

	for true, runs in result:
		correct_instances += np.array(true)
		num_instances += np.array(runs)

	print("Completed Evaluation of", set_file, "with", instance_size, "vertices!")
	for i, name in enumerate(crossovers):
		print(name + ":")
		print("{}/{} runs successful".format(correct_instances[i], num_instances[i]))

def run_instance_helper(path_to_txt, visualize, verbose):
	correct, num_runs = run_instance(path_to_txt, visualize=visualize, verbose=verbose)
	#print(f"running instance: {path_to_txt}")
	#print("{}/{} runs successful".format(correct, num_runs))
	return correct, num_runs

if __name__ == "__main__":
	#Note: Simply modify set when experimenting
	set="setD"
	
	if set=="setA":
		for instance_set in instance_set_A:
			run_set(set, instance_size=instance_set, visualize=False, verbose=False)

	if set=="setB":
		for instance_set in instance_set_B:
			run_set(set, instance_size=instance_set, visualize=False, verbose=False)

	if set=="setC":
		for instance_set in instance_set_C:
			run_set(set, instance_size=instance_set, visualize=False, verbose=False)

	if set=="setD" or set=="setE":
		for instance_set in instance_set_DE:
			run_set(set, instance_size=instance_set, visualize=False, verbose=False)

	#correct, num_runs = run_instance("maxcut-instances/setE/n0000020i00.txt", visualize=False, verbose=True)
	#correct, num_runs = run_instance("maxcut-instances/setE/n0000040i00.txt", visualize=False, verbose=True)
	#correct, num_runs = run_instance("maxcut-instances/setE/n0000080i00.txt", visualize=False, verbose=True)
	#correct, num_runs = run_instance("maxcut-instances/setE/n0000160i00.txt", visualize=False, verbose=True)
	#print("{}/{} runs successful".format(correct, num_runs))
	print("Done")

