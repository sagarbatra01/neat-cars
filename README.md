# NEAT Cars
This repository contains code for performing a car simulation using NEAT (**N**euro**E**volution of **A**ugmenting **T**opologies). NEAT is a genetic algorithm used to generate evolving artificial neural networks by not only learning the weights of the networks, but also their structures.

When the program starts, cars are generated on the track. The cars have different colors depending on the species they belong to, which in turn depends on the genetic similarity of their neural networks. The cars drive using their randomly initialized neural networks until the timer runs out or they all crash. This is when evolution happens: the best performing cars and species (according to the specified metric, by default total travel distance) are selected to survive and reproduce while the rest are removed. Through this cycle of driving and evolving, the population "learns" which neural networks work best for this given task.

![neat-cars-gif-final](https://github.com/sagarbatra01/neat-cars/assets/87910501/fdb36ce3-9208-438b-b71f-0629cddc536b)

## Setup
This program uses Pygame for the graphical interface. 

<details>
  <summary>Show Instructions</summary>
  
  ### 1. Clone repository
  ```
  git clone https://github.com/sagarbatra01/neat-cars.git
  ```

  ### 2. Create and activate a fresh python environment
  ```
  virtualenv neat-cars-env
  neat-cars-env\Scripts\activate
  ```
  
  ### 3. Install Python packages
  ```
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
</details>

## Usage
### 1. Configuration
In the file *config.py* the user can specify the desired configuration for the simulation.

### 2. Track
The user can either use the provided track or create their own as can be seen in the gif above. The track image to be used can be specified in *config.py*. 
The image should be of PNG format. There is no fixed size requirement for the image but if made too large, parts of the image or menu can be outside the screen. The track must be black with white background, and a red line must indicate the starting position and angle of the cars.

### 3. Running the program
Run the *main.py* file:
  ```
  cd neat-cars
  python3 main.py
  ```

## How does it work?
### Driving
During the driving phase, the cars drive using their neural network using a forward pass:

![forward_pass](https://github.com/sagarbatra01/neat-cars/assets/87910501/2eeefc19-3e63-4c77-a599-59774c3880f6)

As can be seen, the inputs to the neural network are the distances of rays from the car to the walls at different angles. Using a forward pass, 2 output values are calculated: the speed and the steering of the car. Note that the neural networks do not change during the driving phase.

### Selection
The selection phase is the first part of the evolution phase.

First each species is sorted based on its species score. The species score is calculated using a score function that the user can redefine in *config.py*. After this a percentage of the species are selected to survive. Out of these species, a percentage of individuals are selected. These percentages can also be modified in the configuration file.

### Reproduction
To maintain the population size, the survivors must reproduce in order to replace the individuals removed in the selection phase. The reproduction works as following:

1. Two random parents are selected from the survivors.
2. Create a child neural network.
3. Copy all of the nodes from the more fit parent to the child.
4. Iterate over all of the edges in the more fit parent. If the edge exists in both parents, copy one of them randomly. Otherwise simply copy the one from the more fit parent.

### Mutation
The new offspring then have a chance to mutate. This is very important as genetic variance is a prerequisite to evolutionary change. In the program there are three types of mutations:

1. Add node: Randomly choose an edge to split in two parts, and insert a new node between these parts.
2. Add edge: Randomly choose 2 nodes and add an edge between them if there is none.
3. Update parameter: Randomly update the bias of a node or the weight of an edge.

The probability of each mutation can be specified by the user in the *config.py* file.

### Speciation
The final part of the evolution process is speciation. The new offspring are now assigned to a suiting species. What is then a suiting species? The method used in this program was compatibility distance: a measure of genetical difference in neural networks described in the paper [Efficient Evolution of Neural Networks](https://nn.cs.utexas.edu/pub-view.php?PubID=114) (2002) by Kenneth O. Stanley and Risto Miikkulainen.

The compatibility distance $\delta$is described with the following equation:

$$ \delta = {c_1 E \over N} + {c_2 D \over N} + {c_3 \bar W}$$

where $E$ and $D$ are the number of excess and disjoint genes (edges) respectively and $\bar W$ is the average weight difference of matching genes.




