# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=consider-using-enumerate, invalid-name, invalid-sequence-index
"""
Cost model optimizer based on simulated annealing
"""

import heapq
import logging
import time

import numpy as np

from ..utils import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob

logger = logging.getLogger("autotvm")


class SimulatedAnnealingOptimizer(ModelOptimizer):
    """parallel simulated annealing optimization algorithm

    Parameters
    ----------
    task: Task
        The tuning task
    n_iter: int
        The number of iterations of simulated annealing
    temp: float or Array of float
        If is a single float, then use a constant temperature.
        If is an Array, then perform linear cooling from temp[0] to temp[1]
    early_stop: int, optional
        Stop iteration if the optimal set do not change in `early_stop` rounds
    log_interval: int, optional
        Print log every `log_interval` iterations
    """

    def __init__(
        self,
        task,
        n_iter=500,
        temp=(1, 0),
        persistent=False,
        parallel_size=128,
        early_stop=50,
        log_interval=50,
    ):
        super(SimulatedAnnealingOptimizer, self).__init__()

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None
        self.history = set()

    def find_maximums(self, model, num, exclusive):
        temp, n_iter, early_stop, log_interval = (
            self.temp,
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size))

        scores = model.predict(points)

        # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([x[1] for x in heap_items])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while (
            k < n_iter
            and k < k_last_modify + early_stop
            and len(self.history) < len(self.task.config_space)
        ):
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                value = random_walk(p, self.dims)
                cnt = 0
                while value in self.history:
                    value = random_walk(value, self.dims)
                    cnt += 1
                    if cnt > 5:
                        break
                self.history.add(value)
                new_points[i] = value
            new_scores = model.predict(new_points)
            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
            ac_index = np.random.random(len(ac_prob)) < ac_prob

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k
            k += 1
            t -= cool
        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items], [x[0] for x in heap_items]


def random_walk(p, dims):
    """random walk as local transition

    Parameters
    ----------
    p: int
        index of the ConfigEntity
    dims: Array of int
        sizes of each dimension

    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # transform to knob form
    old = point2knob(p, dims)
    new = list(old)

    # mutate
    while new == old:
        from_i = np.random.randint(len(old))
        to_v = np.random.randint(dims[from_i])
        new[from_i] = to_v

    # transform to index form
    return knob2point(new, dims)
