# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a wrapper of the CMA-ES algorithm.

This is for users who want to use CMA-ES before we release a pure JAX version.

CMA-ES paper: https://arxiv.org/abs/1604.00772
"""

import sys

import logging
import numpy as np
from typing import Union

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


# Helper classes and functions (NodeGene, ConnectionGene, Genome, initialize_population, select_and_breed, etc.)

class NEATAlgorithm(NEAlgorithm):
    def __init__(self, input_dim: int, output_dim: int, population_size: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.population = initialize_population(population_size, input_dim, output_dim)
        self.current_generation = 0
        self.fitnesses = np.zeros(population_size)
    
    def ask(self) -> List[NEATPolicy]:
        # Return the current population's policies
        policies = []
        for genome in self.population:
            policy = NEATPolicy(self.input_dim, self.output_dim, self.population_size)
            weights = self.genome_to_weights(genome)
            policy.set_weights(weights)
            policies.append(policy)
        return policies

    def tell(self, fitnesses: List[float]):
        # Record the fitnesses for the current population
        self.fitnesses = np.array(fitnesses)

    def genome_to_weights(self, genome: Genome) -> np.ndarray:
        # Convert a genome to a flat array of weights
        weights = [conn.weight for conn in genome.connections]
        return np.array(weights)

    def evolve_population(self):
        # Select, mutate, and crossover to create a new population
        self.population = select_and_breed(self.population, self.fitnesses, self.population_size)
        self.current_generation += 1

    def step(self, n_steps: int = 1):
        for _ in range(n_steps):
            self.evolve_population()

