import numpy as np
import torch
import multiprocessing
import torch.nn.functional as F
from .model_based_tuner import CostModel, FeatureCache
from .model import Encoder, LSTM, MLP
from .. import feature


class OneShotCostModel(CostModel):
    def __init__(self, task, num_threads=None, opt=None):
        super(OneShotCostModel, self).__init__()
        self.model_type = opt.model_type
        self.model_path = opt.model_path
        self.task = task
        self.target = task.target
        self.space = task.config_space
        self.fea_type = opt.feature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type.lower() == "transformer":
            self.model = Encoder(
                embed_size=748,
                num_heads=4,
                num_layers=opt.layer,
                hidden_size=1024,
                dropout=0.25,
            ).to(device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        elif self.model_type.lower() == "lstm":
            self.model = LSTM(748, 1024, 1, 3).to(device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        elif self.model_type.lower() == "mlp":
            self.model = MLP(748, 1024, 1).to(device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        else:
            raise ("invaid model type only support transformer, lstm and mlp !")
        self.feature_extract_func = _extract_curve_feature_index
        self.num_threads = num_threads
        self.pool = None
        self.feature_cache = FeatureCache()
        self.feature_extra_ct = 0
        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""
        self._close_pool()
        # use global variable to pass common arguments
        global _extract_space, _extract_target, _extract_task

        _extract_space = space
        _extract_target = target
        _extract_task = task

        self.pool = multiprocessing.Pool(self.num_threads)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _get_pool(self):
        return self.pool

    def fit(self, xs, ys, plan_size):
        # one-shot tuner only need predict!
        pass

    def fit_log(self, records, plan_size):
        # one-shot tuner only need predict!
        pass

    def predict(self, xs, output_margin=False):
        offset = 100
        pool = self._get_pool()
        features = pool.map(self.feature_extract_func, xs)
        result = np.zeros(len(features))
        mask = []
        filtered = []
        for g in features:
            if type(g) == bool:
                mask.append(False)
            else:
                filtered.append(g)
                mask.append(True)
        if len(filtered) == 0:
            return np.zeros(len(xs))
        filtered = torch.tensor(np.vstack(filtered)).float().cuda().unsqueeze(0)
        with torch.no_grad():
            predict = self.model(filtered).cpu().numpy()
        mask = np.array(mask)
        mask_idx = np.where(mask == True)
        for idx, value in zip(mask_idx[0], predict.squeeze()):
            result[idx] = value + offset
        return result

    def spawn_base_model(self):
        return OneShotCostModel(self.task, self.model_type, self.weight_path, self.embeded_path)

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            # If we are forking, we can pass arguments in globals for better performance
            if multiprocessing.get_start_method(False) == "fork":
                feas = pool.map(self.feature_extract_func, need_extract)
            else:
                args = [(self.space.get(x), self.target, self.task) for x in need_extract]
                feas = pool.map(self.feature_extract_func, args)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea
        ret = []
        for i, ii in enumerate(indexes):
            ret.append(fea_cache[ii])

        return ret

    def __del__(self):
        self._close_pool()


_extract_space = None
_extract_target = None
_extract_task = None


def _extract_curve_feature_index(index, keep_name=False, task_name=None):
    """extract sampled curve feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        with _extract_target:
            sch, args = _extract_task.instantiate(config)
        result = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=30, gpu_filter=True)
        if result == None:
            return False
        else:
            curve = result
        curve_zeros = np.zeros(720)
        curve_zeros[-len(curve) :] = curve
        knobs = config.get_flatten_feature()[:20]
        knob_zeros = np.zeros(20)
        knob_zeros[-len(knobs) :] = knobs
        parameter = np.hstack(
            [
                list(map(lambda x: x.value, args[1].shape)),
                list(map(lambda x: x.value, args[2].shape)),
            ]
        )
        parameter_zeros = np.zeros(8)
        parameter_zeros[-len(parameter) :] = parameter
        return np.hstack([curve_zeros, knob_zeros, parameter_zeros])
    except Exception as e:
        import traceback
        import sys

        traceback.print_exc()
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        return None
