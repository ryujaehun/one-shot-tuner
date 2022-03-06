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
"""Tuner that uses xgboost as cost model"""

from .model_based_tuner import ModelBasedTuner
from .oneshot_cost_model import OneShotCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer
from .oneshot_ga_model_optimizer import GAOptimizer


class OneShotTuner(ModelBasedTuner):
    def __init__(
        self, tasks, num_threads=None, plan_size=32, log_interval=50, optimizer="sa", opt=None
    ):
        cost_model = OneShotCostModel(tasks, num_threads=num_threads, opt=opt)

        if optimizer == "sa":
            optimizer = SimulatedAnnealingOptimizer(
                tasks,
                log_interval=log_interval,
                parallel_size=opt.parallel_size,
                n_iter=opt.n_iter,
                early_stop=opt.early_stop,
            )
        elif optimizer == "ga":
            optimizer = GAOptimizer(tasks, log_interval=log_interval, opt=opt)
        super(OneShotTuner, self).__init__(tasks, cost_model, optimizer, opt.n_parallel, None, opt)

    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(OneShotTuner, self).tune(*args, **kwargs)
        self.cost_model._close_pool()
