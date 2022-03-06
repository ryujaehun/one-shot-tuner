import numpy as np
import heapq
from .model_based_tuner import ModelOptimizer, knob2point, point2knob


class GAOptimizer(ModelOptimizer):
    def __init__(self, task, log_interval=50, opt=None):
        super(GAOptimizer, self).__init__()
        self.opt = opt
        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]
        self.trial = opt.ga_trial
        self.space = self.task.config_space
        self.model = None
        self.visited = set([])
        self.pop_size = opt.pop_size
        self.elite_num = opt.elite_num
        self.mutation_prob = opt.mutation_prob
        assert (
            self.elite_num <= self.pop_size
        ), "The number of elites must be less than population size"
        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0
        self.max_value = 0

        # random initialization
        self.pop_size = min(self.pop_size, len(self.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        for _ in range(self.pop_size):
            tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)
            while knob2point(tmp_gene, self.dims) in self.visited:
                tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)
            self.genes.append(tmp_gene)
            self.visited.add(knob2point(tmp_gene, self.dims))

    def find_maximums(self, model, num, exclusive):
        self.visited.update(exclusive)
        try:
            for trial in range(self.trial):
                point = []
                for p in self.genes:
                    v = knob2point(p, self.dims)
                    point.append(v)
                self.scores = model.predict(point, self.dims)
                # clip under zero value for probs
                self.scores = np.clip(self.scores, 0, None)
                if np.max(self.scores) > self.max_value:
                    self.max_value = np.max(self.scores)
                _score = (self.scores / self.max_value).tolist()
                genes = self.genes + self.elites
                if len(self.elite_scores) > 1:
                    self.elite_scores = (np.array(self.elite_scores) / self.max_value).tolist()
                scores = np.array(_score[: len(self.genes)] + self.elite_scores)
                # reserve elite
                self.elites, self.elite_scores = [], []
                elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num :]
                for ind in elite_indexes:
                    self.elites.append(genes[ind])
                    self.elite_scores.append(scores[ind] * self.max_value)
                # cross over
                indices = np.arange(len(genes))
                probs = scores / np.sum(scores)
                tmp_genes = []
                for _ in range(self.pop_size):
                    p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                    p1, p2 = genes[p1], genes[p2]
                    point = np.random.randint(len(self.dims))
                    tmp_gene = p1[:point] + p2[point:]
                    tmp_genes.append(tmp_gene)
                # mutation
                next_genes = []
                for tmp_gene in tmp_genes:
                    for j, dim in enumerate(self.dims):
                        if np.random.random() < self.mutation_prob:
                            tmp_gene[j] = np.random.randint(dim)
                    if len(self.visited) < len(self.space):
                        while knob2point(tmp_gene, self.dims) in self.visited:
                            j = np.random.randint(len(self.dims))
                            tmp_gene[j] = np.random.randint(
                                self.dims[j]  # pylint: disable=invalid-sequence-index
                            )
                        next_genes.append(tmp_gene)
                        self.visited.add(knob2point(tmp_gene, self.dims))
                    else:
                        break
                self.genes = next_genes
                if len(self.visited) == len(self.space):
                    print(f"find all points # trial {trial} Search space:{len(self.visited)} ")
                    break
        except Exception as e:
            import traceback
            import sys

            traceback.print_exc()
            print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        idx = np.argsort(self.elite_scores)[::-1]
        point = []
        for p in self.elites:
            v = knob2point(p, self.dims)
            point.append(v)
        return (
            np.array(point)[idx].tolist(),
            (np.array(self.elite_scores)[idx] * (self.max_value)).tolist(),
        )
