import numpy as np
import os

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run_instance(inst):
	crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
	for cx in crossovers:
		with open("output-{}.txt".format(cx), "w") as f:
			population_size = 500
			num_evaluations_list = []
			num_runs = 30
			num_success = 0
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				fitness.visualize()
			# genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)
			# best_fitness, num_evaluations = genetic_algorithm.run()
			# if best_fitness == fitness.value_to_reach:
			#	num_success += 1
			# num_evaluations_list.append(num_evaluations)
			print("{}/{} runs successful".format(num_success, num_runs))
			print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
			f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
											  percentiles[2]))

def run_set(set_file):
	path = os.path.join("maxcut-instances", set_file)
	for instance in os.listdir(path):
		if instance.endswith(".txt"):
			run_instance(os.path.join(path, instance))

if __name__ == "__main__":
	run_set("setA")