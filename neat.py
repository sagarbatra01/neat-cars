from enum import Enum

import math
import random as rnd

import config


# Layer = Enum('Layer', ['INPUT', 'HIDDEN', 'OUTPUT'])
class Layer(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Species:
    def __init__(self, members=[]):
        self.members = members
        self.color = (rnd.choices(range(255), k=4))

    def add(self, member):
        self.members.append(member)

    def remove(self, member):
        self.members.remove(member)

    def get_representative(self):
        return self.members[0]


class Node:
    def __init__(self, id, type=Layer.HIDDEN):
        self.id = id
        self.value = 0
        self.type = type


class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = True


class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.nodes = {}
        self.edges = {}
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness = 0
        self.input_nodes = {}
        self.layer_size = [num_inputs, 0, num_outputs]

        # Initialize input nodes
        for i in range(num_inputs):
            input_node = Node(id=i, type=Layer.INPUT)
            self.nodes[i] = input_node
            self.input_nodes[i] = input_node

        # Initialize output nodes
        for i in range(num_outputs):
            output_node_id = i + num_inputs  # Ensure unique IDs for output nodes
            output_node = Node(id=output_node_id, type=Layer.OUTPUT)
            self.nodes[output_node_id] = output_node

        # Create edges between input and output nodes
        for i in range(num_inputs):
            for j in range(num_inputs, num_inputs + num_outputs):
                edge = Edge(i, j, rnd.uniform(-1, 1))
                self.edges[(i, j)] = edge

    def __str__(self):
        info = f"Neural Network Information:\n"
        info += f"Input Nodes: {self.num_inputs}\n"
        info += f"Output Nodes: {self.num_outputs}\n"
        info += f"Fitness: {self.fitness}\n"
        info += "Nodes:\n"
        for node_id, node in self.nodes.items():
            info += f"  Node ID: {node_id}, Type: {node.type}\n"
        info += "Edges:\n"
        for (from_node, to_node), edge in self.edges.items():
            info += f"  From Node: {from_node}, To Node: {to_node}, Weight: {edge.weight}, Enabled: {edge.enabled}\n"
        return info

    def feed_forward(self, inputs):
        values = {node_id: 0.0 for node_id in self.nodes}

        for input_id, input_value in enumerate(inputs):
            values[input_id] = input_value

        for edge_id, edge in self.edges.items():
            if edge.enabled:
                from_node_id = edge.from_node
                to_node_id = edge.to_node
                weight = edge.weight
                weighted_sum = values[from_node_id] * weight

                values[to_node_id] += weighted_sum

        output = (values[id]
                  for id in self.nodes if self.nodes[id].type == Layer.OUTPUT)
        return output


class NEAT:
    def __init__(self, num_inputs, num_outputs, population_size):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.generation = 1
        self.population = []

        for _ in range(population_size):
            individual = NeuralNetwork(num_inputs, num_outputs)
            self.population.append(Species([individual]))

    def get_individuals(self):
        return [individual for species in self.population for individual in species.members]

    def add_node(self, neural_network):
        enabled_edges = [
            edge for edge in neural_network.edges.values() if edge.enabled]
        if enabled_edges:
            edge_to_split = rnd.choice(enabled_edges)

            # Disable the original edge
            edge_to_split.enabled = False

            # Create a new node and two new edges
            new_node_id = max(neural_network.nodes) + 1
            new_edge1 = Edge(edge_to_split.from_node, new_node_id, 1.0)
            new_edge2 = Edge(new_node_id, edge_to_split.to_node,
                             edge_to_split.weight)

            # Add the new node and edges to the neural network
            neural_network.nodes[new_node_id] = Node(new_node_id)
            neural_network.layer_size[Layer.HIDDEN.value] += 1

            # print("added", new_node_id)
            neural_network.edges[(edge_to_split.from_node,
                                  new_node_id)] = new_edge1
            neural_network.edges[(
                new_node_id, edge_to_split.to_node)] = new_edge2

    def add_edge(self, nn):
        from_node = rnd.choice(
            [id for id, node in nn.nodes.items() if node.type != Layer.OUTPUT])
        to_node = rnd.choice(
            [id for id, node in nn.nodes.items() if node.type != Layer.INPUT])

        while from_node == to_node:
            to_node = rnd.choice(
                [id for id, node in nn.nodes.items() if node.type != Layer.INPUT])

        # Check if the edge already exists, if not, create it
        if (from_node, to_node) not in nn.edges and from_node != to_node:
            new_edge = Edge(from_node, to_node, rnd.uniform(-1, 1))
            nn.edges[(from_node, to_node)] = new_edge

    def update_weight(self, neural_network):
        if neural_network.edges:
            edge_to_update = rnd.choice(list(neural_network.edges.values()))
            edge_to_update.weight = rnd.uniform(-1, 1)

    def create_offspring(self):
        offspring = NeuralNetwork(self.num_inputs, self.num_outputs)
        # self.mutate(offspring)
        return offspring

    def genetic_difference(self, individual1, individual2):
        """Calculate the compatability distance."""
        disjoint_edges = 0
        matching_edges = 0
        weight_difference = 0
        matched_edges_set = set()

        for edge1 in individual1.edges.values():
            match_found = False
            for edge2 in individual2.edges.values():
                if edge1.from_node == edge2.from_node and edge1.to_node == edge2.to_node:
                    matching_edges += 1
                    weight_difference += abs(edge1.weight -
                                             edge2.weight)
                    match_found = True
                    matched_edges_set.add(edge2)
                    break

            if not match_found:
                disjoint_edges += 1

        excess_edges = len(individual2.edges) - len(matched_edges_set)
        num_edges = max(len(individual1.edges), len(individual2.edges), 1)

        c1, c2, c3 = 1, 1, 1  # Adjustable constants
        compatibility_distance = c1 * excess_edges/num_edges \
            + c2 * disjoint_edges/num_edges \
            + c3 * weight_difference

        return compatibility_distance

    def mutate(self, neural_network):
        if rnd.random() < config.MUTATION_RATE_ADD_NODE:
            self.add_node(neural_network)

        if rnd.random() < config.MUTATION_RATE_ADD_EDGE:
            self.add_edge(neural_network)

        if rnd.random() < config.MUTATION_RATE_UPDATE_WEIGHT:
            self.update_weight(neural_network)

    def crossover(self, parent1, parent2):
        offspring = NeuralNetwork(parent1.num_inputs, parent1.num_outputs)
        fittest_parent = max([parent1, parent2], key=lambda x: x.fitness)

        # Copy nodes from fittest parent to offspring
        for id in fittest_parent.nodes:
            offspring.nodes[id] = Node(id, fittest_parent.nodes[id].type)

        for id, edge in fittest_parent.edges.items():
            if edge.enabled:
                # If the edge exists in both parents, choose one randomly
                if fittest_parent.nodes[edge.from_node].id in parent1.nodes and fittest_parent.nodes[edge.to_node].id in parent2.nodes:
                    if rnd.random() < 0.5:
                        new_edge = parent1.edges[id]
                    else:
                        new_edge = parent2.edges[id]
                else:
                    # If the edge only exists in the fittest parent, choose it
                    new_edge = edge

            # Add the chosen edge to the offspring
            offspring.edges[id] = new_edge
        return offspring

    def speciate(self, unassigned):
        for individual in unassigned:
            assigned_species = False

            for species in self.population:
                species_representative = species.get_representative()

                difference = self.genetic_difference(
                    individual, species_representative)

                if difference < config.COMPATIBILITY_THRESHOLD:
                    species.add(individual)
                    assigned_species = True
                    break

            if not assigned_species:
                self.population.append(Species([individual]))

    def select(self):
        # Sort species in the order of their maximal fitness
        sorted_species = sorted(self.population, key=lambda species: max(
            m.fitness for m in species.members), reverse=True)

        for species in sorted_species[:math.ceil(len(sorted_species) * config.SPECIES_SURVIVAL_RATE)]:
            survivors = math.ceil(len(species.members) * config.OFFSPRING_RATE)
            species.members.sort(key=lambda m: m.fitness, reverse=True)
            species.members = species.members[:survivors]
            if not species.members:
                self.population.remove(species)

        for species in sorted_species[math.ceil(len(sorted_species) * config.SPECIES_SURVIVAL_RATE):]:
            self.population.remove(species)

    def reproduce(self):
        new_offspring = []
        while len(self.get_individuals()) + len(new_offspring) < self.population_size:
            new_individual = self.create_offspring()
            new_offspring.append(new_individual)
        return new_offspring

    def evolve(self, time):
        # self.print_fitnesses()
        self.generation += 1
        fitnesses = [m.fitness/(time/1000) for m in self.get_individuals()]
        self.select()
        new_offspring = self.reproduce()
        self.speciate(new_offspring)
        for species in self.population:
            for nn in species.members:
                self.mutate(nn)
                nn.fitness = 0
        return self.generation, fitnesses, self.population

    def print_fitnesses(self):
        for individual in self.get_individuals():
            print(individual.fitness)
