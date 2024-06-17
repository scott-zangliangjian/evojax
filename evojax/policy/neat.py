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

import logging
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

class NodeGene:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type

class ConnectionGene:
    def __init__(self, innovation_number, from_node, to_node, weight, enabled=True):
        self.innovation_number = innovation_number
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

class Genome:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections

def initialize_population(population_size, input_dim, output_dim):
    population = []
    for _ in range(population_size):
        nodes = [NodeGene(i, 'input') for i in range(input_dim)] + [NodeGene(input_dim + i, 'output') for i in range(output_dim)]
        connections = [ConnectionGene(i, random.randint(0, input_dim-1), input_dim + random.randint(0, output_dim-1), random.uniform(-1, 1)) for i in range(input_dim * output_dim)]
        genome = Genome(nodes, connections)
        population.append(genome)
    return population

def decode_genome(weights, input_dim, output_dim):
    connections = []
    for i, weight in enumerate(weights):
        from_node = i // output_dim
        to_node = i % output_dim + input_dim
        connections.append(ConnectionGene(i, from_node, to_node, weight))
    nodes = [NodeGene(i, 'input') for i in range(input_dim)] + [NodeGene(input_dim + i, 'output') for i in range(output_dim)]
    return Genome(nodes, connections)

class NEATPolicy(PolicyNetwork):
    def __init__(self, input_dim, output_dim, population_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.population = initialize_population(population_size, input_dim, output_dim)
        self.current_genome = None
        self.current_weights = None

    def set_weights(self, weights):
        self.current_genome = decode_genome(weights, self.input_dim, self.output_dim)
        self.current_weights = weights

    def feedforward(self, genome, inputs):
        node_values = {node.node_id: 0.0 for node in genome.nodes}
        for i, input_value in enumerate(inputs):
            node_values[i] = input_value

        for conn in genome.connections:
            if conn.enabled:
                node_values[conn.to_node] += node_values[conn.from_node] * conn.weight

        output_values = jnp.array([node_values[node.node_id] for node in genome.nodes if node.node_type == 'output'])
        return output_values

    def get_actions(self, params, obs):
        if self.current_genome is None:
            raise ValueError("Weights not set. Call set_weights() before get_actions().")

        obs = jnp.asarray(obs)
        return jax.vmap(lambda x: self.feedforward(self.current_genome, x))(obs)

    def reset(self):
        pass

    def get_params(self):
        return self.current_weights

