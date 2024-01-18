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
The image should be of PNG format. There is no fixed size requirement for the image but if made too large, parts of the image or menu can be outside the screen.

### 3. Running the program
Run the *main.py* file:
  ```
  cd neat-cars
  python3 main.py
  ```

## How does it work?
### Driving


### Selection


### Reproduction


### Mutation


### Speciation

