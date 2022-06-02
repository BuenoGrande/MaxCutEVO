import numpy as np
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

import Individual
from Utils import ValueToReachFoundException

class FitnessFunction:
	def __init__( self ):
		self.dimensionality = 1 
		self.number_of_evaluations = 0
		self.value_to_reach = np.inf

	def evaluate( self, individual: Individual ):
		self.number_of_evaluations += 1
		if individual.fitness >= self.value_to_reach:
			raise ValueToReachFoundException(individual)


class OneMax(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.value_to_reach = dimensionality

	def evaluate( self, individual: Individual ):
		individual.fitness = np.sum(individual.genotype)
		super().evaluate(individual)

class DeceptiveTrap(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.trap_size = 5
		assert dimensionality % self.trap_size == 0, "Dimensionality should be a multiple of trap size"
		self.value_to_reach = dimensionality

	def trap_function( self, genotype ):
		assert len(genotype) == self.trap_size
		k = self.trap_size
		bit_sum = np.sum(genotype)
		if bit_sum == k:
			return k
		else:
			return k-1-bit_sum

	def evaluate( self, individual: Individual ):
		num_subfunctions = self.dimensionality // self.trap_size
		result = 0
		for i in range(num_subfunctions):
			result += self.trap_function(individual.genotype[i*self.trap_size:(i+1)*self.trap_size])
		individual.fitness = result
		super().evaluate(individual)

class MaxCut(FitnessFunction):
	def __init__( self, instance_file ):
		super().__init__()
		self.edge_list = []
		self.weights = {}
		self.adjacency_list = {}
		self.G = nx.Graph()
		self.clusters = []
		self.read_problem_instance(instance_file)
		self.read_value_to_reach(instance_file)
		self.preprocess()

	def preprocess( self ):
		pass

	def read_problem_instance( self, instance_file ):
		with open( instance_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.dimensionality = int(first_line[0])
			number_of_edges = int(first_line[1])
			for line in lines[1:]:
				splt = line.split()
				v0 = int(splt[0])-1
				v1 = int(splt[1])-1
				assert( v0 >= 0 and v0 < self.dimensionality )
				assert( v1 >= 0 and v1 < self.dimensionality )
				w = float(splt[2])
				self.edge_list.append((v0,v1))
				self.weights[(v0,v1)] = w
				self.weights[(v1,v0)] = w
				self.G.add_edge(v0,v1, weight=w)
				if( v0 not in self.adjacency_list ):
					self.adjacency_list[v0] = []
				if( v1 not in self.adjacency_list ):
					self.adjacency_list[v1] = []
				self.adjacency_list[v0].append(v1)
				self.adjacency_list[v1].append(v0)
			assert( len(self.edge_list) == number_of_edges )
	
	def read_value_to_reach( self, instance_file ):
		bkv_file = instance_file.replace(".txt",".bkv")
		with open( bkv_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.value_to_reach = float(first_line[0])

	def get_weight( self, v0, v1 ):
		if( not (v0,v1) in self.weights ):
			return 0
		return self.weights[(v0,v1)]

	def get_degree( self, v ):
		return len(self.adjacency_list[v])

	def evaluate( self, individual: Individual ):
		result = 0
		for e in self.edge_list:
			v0, v1 = e
			w = self.weights[e]
			if( individual.genotype[v0] != individual.genotype[v1] ):
				result += w

		individual.fitness = result
		super().evaluate(individual)

	def visualize(self):
		pos = nx.spring_layout(self.G, seed=42)  # For better example looking
		nx.draw(self.G, pos, with_labels=True)
		labels = [self.G.edges[e]['weight'] for e in self.G.edges]
		nx.draw_networkx_edges(self.G, pos, width=labels)

		plt.show()

	def visualize_individual(self, individual):
		pos = nx.spring_layout(self.G, seed=42)  # For better example looking
		nx.draw(self.G, pos, node_color=individual.genotype[self.G.nodes], with_labels=True)
		labels = [self.G.edges[e]['weight'] for e in self.G.edges]
		nx.draw_networkx_edges(self.G, pos, width=labels)

		plt.show()

	def compute_node_potential(self, individual: Individual):
		results = np.zeros(len(individual.genotype))
		for e in self.edge_list:
			v0, v1 = e
			w = self.weights[e]
			sign = -1 if individual.genotype[v0] != individual.genotype[v1] else 1
			results[v0] += sign * w
			results[v1] += sign * w
		return results

	def choose_best_individuals(self, individual_a: Individual, individual_b: Individual):
		if np.random.random() < 0.1:
			k = [np.random.random() < 0.3 for _ in individual_a.genotype]
		else:
			k = [False for _ in individual_a.genotype]

		# potential is large if a node switch would improve the max-cut
		potential_a = self.compute_node_potential(individual_a)

		return np.array(k) ^ (self.get_non_interfering_nodes(potential_a, individual_a, individual_b) == 1)

	def get_non_interfering_nodes(self, potential, individual_a, individual_b):
		out = np.ones(potential.shape)*-1
		potential_nodes = np.argsort(potential)
		for idx in reversed(potential_nodes):
			if out[idx] == -1 and \
					potential[idx] > 0 and \
					individual_a.genotype[idx] != individual_b.genotype[idx]:
				out[idx] = 1
				for neighboring_idx in self.adjacency_list[idx]:
					out[neighboring_idx] = 0
			else:
				out[idx] = 0
		return out

	def choose_best_cluster(self, individual_a: Individual, individual_b: Individual):
		if not self.clusters:
			self.compute_clusters()

		return self.evaluate_clusters(individual_a, individual_b)

	def compute_clusters(self):
		l = len(self.G.nodes)
		visited = np.zeros(l)

		for v0 in range(l):
			if not visited[v0]:
				neighborhood = [v0] + self.adjacency_list[v0]
				for v1 in self.adjacency_list[v0]:
					neighborhood += self.adjacency_list[v1]

				counter = Counter(neighborhood)
				cluster = []
				[cluster.append(v) for v in neighborhood if counter[v] != 1 and v not in cluster]

				assert len(cluster) == 5
				for v in cluster:
					visited[v] = 1
				self.clusters.append(cluster)

	def evaluate_clusters(self, individual_a, individual_b):
		out = np.zeros(len(individual_a.genotype))
		for cluster in self.clusters:
			if (individual_a.genotype[cluster] == individual_b.genotype[cluster]).all():
				continue

			if self.evaluate_cluster(cluster, individual_b) > self.evaluate_cluster(cluster, individual_a):
				if np.random.random() < 0.9:
					out[cluster] = True
				else:
					out[cluster] = np.random.choice([0, 1], len(cluster), p=[0.1, 0.9])
			else:
				out[cluster] = np.random.choice([0, 1], len(cluster), p=[0.9, 0.1])
		return out

	def evaluate_cluster(self, cluster, individual):
		out = 0
		for i in range(len(cluster)-1):
			for j in range(i+1, len(cluster)):
				if individual.genotype[cluster[i]] != individual.genotype[cluster[j]]:
					out += self.get_weight(cluster[i], cluster[j])
		return out
