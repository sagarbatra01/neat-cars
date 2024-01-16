from enum import Enum
import math
import random as rnd
import copy

import config as cf


class Layer(Enum):
    """
    Represents the layers in the neural networks.

    This class serves as an enumerator to simplify calculations based 
    on the layers' indices.
    """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class Species:
    """
    Represents a species in the population.

    This class acts as a data structure to represent a species, 
    providing attributes and methods to simplify the representation.

    Attributes:
    - members (list): A list of all the neural networks in this species.
    - color (tuple): A tuple representing the RGBA color of the species.

    Methods:
    - add(member): Add the member to the list of members.

    - remove(member): Remove the member from the list of members.

    - get_representative(): Return the member to represent the species 
    when determing similarity between another individual and this species.
    """

    def __init__(self, members=None):
        self.members = members if members else []
        self.color = (rnd.choices(range(255), k=4))

    def add(self, member):
        self.members.append(member)

    def remove(self, member):
        self.members.remove(member)

    def get_representative(self):
        """Return the member to represent the species when determing 
        similarity between another individual and this species.

        Returns:
            NeuralNetwork: The neural network to represent this species.
        """
        return max(self.members, key=lambda m: m.fitness)


class Node:
    """
    Represents a node in a neural network.

    This class serves as a building block to represent nodes in the 
    NeuralNetwork class.

    Attributes:
    - id (int): An integer representing the Node identifier.
    - type (Layer): A Layer representing the Node's layer.
    - value (float): A temporary value for the Node during feed forward.
    - bias (float): The Node's bias parameter.
    """

    def __init__(self, id, type=Layer.HIDDEN, bias=rnd.uniform(-1, 1)):
        self.id = id
        self.type = type
        self.value = 0
        self.bias = bias

    def __deepcopy__(self, memo):
        return self.__class__(self.id, self.type, self.bias)


class Edge:
    """
    Represents an edge in a neural network.

    This class serves as a building block to represent edges in the 
    NeuralNetwork class.

    Attributes:
    - from_node (int): An identifier of the node in which the edge starts.
    - to_node (int): An identifier of the node in which the edge ends.
    - weight (float): The weight of the edge.
    - enabled (bool): A boolean representing whether or not the edge is enabled.
    """

    def __init__(self, from_node, to_node, weight, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

    def __deepcopy__(self, memo):
        return self.__class__(self.from_node, self.to_node, self.weight, self.enabled)


class NeuralNetwork:
    """
    Represents the neural network for an individual.

    This is a custom class for a 2-3 layer neural network.

    Attributes:
    - nodes (dict): A dictionary mapping all node IDs to a node.
    - edges (dict): A dictionary mapping start and end node IDs to an edge.
    - fitness (float): The fitness of the individual of this neural network.
    - layer_size (bool): A list where index i represents the size of layer i.

    Methods:
    - feed_forward(inputs): Return the result of a forward pass in the network.

    - mutate(): Mutate the neural network with the assigned probabilities.

    - add_node(): Add a Node to the neural network in the middle of an existing Edge

    - add_edge(): Add an edge between two randomly chosen nodes in the network.

    - update_param(): Update a random parameter in the Neural Network, 
        either an edge's weight or a node's bias.
    """

    def __init__(self, num_inputs, num_outputs):
        self.nodes = {}
        self.edges = {}
        self.fitness = 0
        self.layer_size = [num_inputs, 0, num_outputs]

        # Initialize input nodes:
        for i in range(num_inputs):
            input_node = Node(id=i, type=Layer.INPUT)
            self.nodes[i] = input_node

        # Initialize output nodes:
        for i in range(num_outputs):
            output_node_id = i + num_inputs
            output_node = Node(id=output_node_id, type=Layer.OUTPUT)
            self.nodes[output_node_id] = output_node

        # Create edges between input and output nodes:
        for i in range(num_inputs):
            for j in range(num_inputs, num_inputs + num_outputs):
                edge = Edge(i, j, rnd.uniform(-1, 1))
                self.edges[(i, j)] = edge

    def __str__(self):
        info = f"Neural Network Information:\n"
        info += f"Input Nodes: {self.layer_size[Layer.INPUT.value]}\n"
        info += f"Output Nodes: {self.layer_size[Layer.OUTPUT.value]}\n"
        info += f"Fitness: {self.fitness}\n"
        info += "Nodes:\n"
        for node_id, node in self.nodes.items():
            info += f"  Node ID: {node_id}, Type: {node.type} Bias: {node.bias}\n"
        info += "Edges:\n"
        for (from_node, to_node), edge in self.edges.items():
            info += f"  From Node: {from_node}, To Node: {to_node}, Weight: {edge.weight}, Enabled: {edge.enabled}\n"
        return info

    def feed_forward(self, inputs):
        """Perform a forward pass in the network and return the result.

        Args:
            inputs (list): List of the new input values to the input layer.

        Returns:
            list: List of output values that are the results of the forward pass.
        """
        values = {node_id: node.bias for node_id, node in self.nodes.items()}

        for input_id, input_value in enumerate(inputs):
            values[input_id] += input_value

        for _, edge in self.edges.items():
            if edge.enabled:
                from_node_id = edge.from_node
                to_node_id = edge.to_node
                weight = edge.weight
                weighted_sum = values[from_node_id] * weight
                values[to_node_id] += weighted_sum

        return (values[id] for id in self.nodes if self.nodes[id].type == Layer.OUTPUT)

    def mutate(self):
        """Mutate the neural network with the assigned probabilities.

        The types of mutations:
            Add node: Add a node in the middle of an existing edge.
            Add edge: Add an edge between two currently existing nodes.
            Update param: Update a random weight or bias in the network.
        """
        if rnd.random() < cf.MUTATION_RATE_ADD_NODE:
            self.add_node()

        if rnd.random() < cf.MUTATION_RATE_ADD_EDGE:
            self.add_edge()

        if rnd.random() < cf.MUTATION_RATE_UPDATE_PARAM:
            self.update_param()

    def add_node(self):
        """Add a node to the neural network in the middle of an existing Edge."""
        enabled_edges = [edge for edge in self.edges.values() if edge.enabled]
        if enabled_edges:
            edge = rnd.choice(enabled_edges)
            edge.enabled = False

            # Create a new node and two new edges
            new_node_id = max(self.nodes) + 1
            new_edge1 = Edge(edge.from_node, new_node_id, 1.0)
            new_edge2 = Edge(new_node_id, edge.to_node, edge.weight)

            # Add the new node and edges to the neural network
            self.nodes[new_node_id] = Node(new_node_id)
            self.layer_size[Layer.HIDDEN.value] += 1
            self.edges[(edge.from_node, new_node_id)] = new_edge1
            self.edges[(new_node_id, edge.to_node)] = new_edge2

    def add_edge(self):
        """Add an edge between two randomly chosen nodes in the network."""
        n1 = []
        n2 = []
        for id, node in self.nodes.items():
            if node.type != Layer.OUTPUT:
                n1.append(id)
            if node.type != Layer.INPUT:
                n2.append(id)

        from_node, to_node = rnd.choice(n1), rnd.choice(n2)
        while from_node == to_node:
            to_node = rnd.choice(n2)

        # Check if the edge already exists, if not, create it
        if (from_node, to_node) not in self.edges and from_node != to_node:
            new_edge = Edge(from_node, to_node, rnd.uniform(-1, 1))
            self.edges[(from_node, to_node)] = new_edge

    def update_param(self):
        """Update a random parameter in the neural network, 
        either an edge's weight or a node's bias.
        """
        param_type = rnd.choice(["weight", "bias"])
        if param_type == "weight" and self.edges:
            edge_to_update = rnd.choice(list(self.edges.values()))
            edge_to_update.weight += rnd.gauss(0, 1)
        if param_type == "bias" and self.nodes:
            node_to_update = rnd.choice(list(self.nodes.values()))
            node_to_update.bias += rnd.gauss(0, 1)


class NEAT:
    """
    Class to simplify the implementation of NeuroEvolution of Augmenting Topologies.

    Attributes:
    - generation (int): An integer representing the current generation number.
    - population (list): A list containing all the current Species.

    Methods:
    - get_individuals(): Return all the Neural Networks in the population.

    - genetic_difference(): Return the genetic difference between two 
        individuals based on compatability distance

    - speciate(): Assign a species to the individual based on the genetic compability.

    - select(): Select the survivors of the current generation.

    - crossover(): Apply crossover on the given parents.

    - reproduce(): Let the surviving population reproduce by applying crossover 
        on two randomly selected parents.

    - evolve():
        Evolve the population to a new generation by selecting the
        best performing species and offspring and replacing the others
        with neural networks similar to the well-performing ones.
    """

    def __init__(self):
        self.generation = 1
        self.population = []
        for _ in range(cf.POPULATION_SIZE):
            individual = NeuralNetwork(cf.NUM_INPUTS, cf.NUM_OUTPUTS)
            self.population.append(Species([individual]))

    def get_individuals(self):
        return [individual for species in self.population for individual in species.members]

    def genetic_difference(self, nn1, nn2):
        """Return the genetic difference between two Neural Networks based 
        on compatability distance (see O. Stanley, K. Miikkulainen Risto.
        2002. Evolving Neural Networks through Augmenting Topologies).

        Args:
            nn1 (NeuralNetwork): The first neural network.
            nn2 (NeuralNetwork): The second neural network.

        Returns:
            float: The compatibility distance between the two networks.
        """
        disjoint_edges = 0
        matching_edges = 0
        weight_difference = 0
        matched_edges_set = set()

        for e1 in nn1.edges.values():
            match_found = False
            for e2 in nn2.edges.values():
                if e1.from_node == e2.from_node and e1.to_node == e2.to_node:
                    matching_edges += 1
                    weight_difference += abs(e1.weight - e2.weight)
                    matched_edges_set.add(e2)
                    match_found = True
                    break

            if not match_found:
                disjoint_edges += 1

        excess_edges = len(nn2.edges) - len(matched_edges_set)
        num_edges = max(len(nn1.edges), len(nn2.edges), 1)

        c1, c2, c3 = cf.C1, cf.C2, cf.C3
        compatibility_distance = \
            c1 * excess_edges/num_edges + \
            c2 * disjoint_edges/num_edges + \
            c3 * weight_difference

        return compatibility_distance

    def speciate(self, individual):
        """Assign a species to the individual based on the genetic compability.

        Args:
            individual (NeuralNetwork): The neural network to assign a species to.
        """
        assigned_species = False
        for species in self.population:
            species_representative = species.get_representative()
            difference = self.genetic_difference(
                individual, species_representative)
            if difference < cf.COMPATIBILITY_THRESHOLD:
                species.add(individual)
                assigned_species = True
                break
        if not assigned_species:
            self.population.append(Species([individual]))

    def select(self):
        """Select the survivors of the current generation."""
        # Order each species according to the specified score function:
        ordered = sorted(self.population, key=cf.species_score, reverse=True)

        # Try to select the best species and individuals:
        num_surviving_species = math.ceil(len(ordered)*cf.SPECIES_SURVIVAL)
        for species in ordered[:num_surviving_species]:
            survivors = math.ceil(len(species.members)*cf.INDIVIDUAL_SURVIVAL)
            species.members.sort(key=lambda m: m.fitness, reverse=True)
            species.members = species.members[:survivors]

            # Remove the species if it is empty:
            if not species.members:
                self.population.remove(species)

        # Remove the species that did not survive:
        for species in ordered[num_surviving_species:]:
            self.population.remove(species)

    def crossover(self, more_fit_parent, less_fit_parent):
        """Apply crossover on the given parents.

        Args:
            more_fit_parent (NeuralNetwork): The parent network with higher fitness.
            less_fit_parent (NeuralNetwork): The parent network with lower fitness.

        Returns:
            NeuralNetwork: The child that was produced through crossover.
        """
        child = NeuralNetwork(cf.NUM_INPUTS, cf.NUM_OUTPUTS)
        for node_id, node in more_fit_parent.nodes.items():
            child.nodes[node_id] = copy.deepcopy(node)
        child.layer_size = copy.deepcopy(more_fit_parent.layer_size)

        for id in more_fit_parent.edges:
            if id in less_fit_parent.edges:
                child.edges[id] = copy.deepcopy(
                    rnd.choice([more_fit_parent.edges[id]]))
            else:
                child.edges[id] = copy.deepcopy(more_fit_parent.edges[id])
        return child

    def reproduce(self):
        """Let the surviving population reproduce by applying crossover 
        on two randomly selected parents.

        Returns:
            list: A list of all the new offspring produced through reproduction.
        """
        new_offspring = []
        while len(self.get_individuals()) + len(new_offspring) < cf.POPULATION_SIZE:
            p1, p2 = rnd.sample(self.get_individuals(), 2)
            if p2.fitness > p1.fitness:
                p1, p2 = p2, p1
            new_offspring.append(self.crossover(p1, p2))
        return new_offspring

    def evolve(self):
        """Evolve the population to a new generation by selecting the
        best performing species and offspring and replacing the others
        with neural networks similar to the well-performing ones.
        """
        self.select()
        new_offspring = self.reproduce()
        for nn in new_offspring:
            nn.mutate()
            self.speciate(nn)
        self.generation += 1
        for nn in self.get_individuals():
            nn.fitness = 0
