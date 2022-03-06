import argparse
import json
import logging
import os
import numpy as np
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, RandomTuner

import utils
import time

parser = argparse.ArgumentParser(prog="autotune-llvm")
parser.add_argument("--target", type=str, default="cuda", help="choose optmization target")
parser.add_argument("--save", default="result", type=str, help="save folder path")
parser.add_argument("--n_parallel", type=int, default=64, help="measure size")
parser.add_argument("--batch", type=int, default=1, help="batch size")
parser.add_argument("--n_trial", type=int, default=1000, help="# of sample per task")
parser.add_argument(
    "--layout",
    type=str,
    default="NCHW",
    choices=["NCHW", "NHWC"],
    help="layout of input and operation",
)
parser.add_argument(
    "-p", "--prior_based_sampling", action="store_true", help="use prior based task sampling?"
)
parser.add_argument(
    "-e",
    "--exploration_based_sampling",
    action="store_true",
    help="use exploration based code sampling?",
)
np.random.seed(np.random.randint(10000000))
opt = parser.parse_args()
save_dir = f"{os.path.dirname(os.path.realpath(__file__))}"

print(f"option", opt)


save_dir = os.path.join(save_dir, opt.layout, str(opt.batch), opt.save)

opt.save_dir = save_dir
os.makedirs(save_dir, exist_ok=True)

print(f"Creating folder {format(save_dir)}")
dtype = "float32"
batch_size = opt.batch
model_name = "random"
log_file = f"{save_dir}/{model_name}.log"
best_log_file = f"{save_dir}/best_{model_name}.log"
num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)
target = tvm.target.Target(opt.target)

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(vars(opt), f, indent=2)

logging.info(
    "Saving configuration file in `{0}`".format(
        os.path.abspath(os.path.join(save_dir, "config.json"))
    )
)

tuning_option = {
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10, n_parallel=opt.n_parallel),
        runner=autotvm.LocalRunner(number=10, repeat=2, timeout=10),
    ),
    "n_trial": opt.n_trial,
}


def tune_tasks(
    tasks,
    measure_option,
    n_trial=64,
    early_stopping=None,
):
    # create tmp log file

    for i, tsk in enumerate(tasks):

        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        opt.task = i + 1
        opt.count = 1
        opt.trial = 1
        os.makedirs(f"{opt.save_dir}/{opt.task}", exist_ok=True)

        print(f"task : {tsk.name} space {len(tsk.config_space)}")
        # create tuner

        tsk_trial = min(n_trial, len(tsk.config_space))
        if opt.prior_based_sampling:
            tuner_obj = XGBTuner(
                tsk,
                feature_type="curve",
                loss_type="reg",
                plan_size=opt.n_parallel,
                optimizer="sa",
            )
        else:
            tuner_obj = RandomTuner(tsk, (0, tsk_trial))
        cb = [
            autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
            autotvm.callback.log_to_file(f"{save_dir}/{model_name}.log"),
        ]
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=cb,
            opt=opt,
        )
    autotvm.record.pick_best(log_file, best_log_file)


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    try:
        print("Extract tasks...")
        tasks = utils.get_random_data(
            save_dir, 10, opt.exploration_based_sampling, opt.batch, opt.layout
        )
        print(tasks)
        # run tuning tasks
        print("Tuning...")
        start = time.time()
        tune_tasks(tasks, **tuning_opt)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Total Data Generation time {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )

    except Exception as e:
        import traceback
        import sys

        traceback.print_exc()
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        sys.exit("fatal error")


if __name__ == "__main__":
    tune_and_evaluate(tuning_option)
