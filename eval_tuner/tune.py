import argparse
import json
import numpy as np
import logging
import os
import time

parser = argparse.ArgumentParser(prog="autotune-llvm")
parser.add_argument("--network", type=str, default="resnet-18", help="choose network")
parser.add_argument("--target", type=str, default="cuda", help="choose optmization target")
parser.add_argument("--batch", type=int, default=8, help="choose batch size")
parser.add_argument("--thread", type=int, default=16, help="choose num of threads")
parser.add_argument("--n_trial", type=int, default=64, help="choose num of trial")
parser.add_argument("--save", default="ONESHOT", type=str, help="save path")
parser.add_argument("--layout", default="NCHW", type=str, help="layout")
#################### neural predictor #############################
parser.add_argument("--optimizer", type=str, default="ga", help="choose optimizer")
parser.add_argument("--n_parallel", type=int, default=32, help="measure size")
parser.add_argument("--layer", type=int, default=3, help="size of layer")
#################### SA  #############################
parser.add_argument("--parallel_size", type=int, default=512, help="exploration module batch size")
parser.add_argument("--n_iter", type=int, default=160, help="how to trial sa step")
parser.add_argument("--early_stop", type=int, default=50, help="early stop step")
#################### GA  #############################
parser.add_argument("--pop_size", type=int, default=256, help="population size")
parser.add_argument("--elite_num", type=int, default=96, help="elite number")
parser.add_argument("--mutation_prob", type=float, default=0.1, help="mutation prob")
parser.add_argument("--ga_trial", type=int, default=120, help="ga trial")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--model_type", type=str, default="transformer")
parser.add_argument("--feature", type=str, default="curve")
np.random.seed(np.random.randint(1000000))
opt = parser.parse_args()


def tune_tasks(
    tasks, measure_option, tuner="sa", n_trial=32, early_stopping=None, log_filename="tuning.log"
):
    from tvm import autotvm
    from tvm.autotvm.tuner import OneShotTuner

    start = time.time()

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        opt.task = i + 1
        opt.count = 1
        opt.trial = 1
        os.makedirs(f"{opt.save_dir}/{opt.task}", exist_ok=True)
        print(f"task : {tsk.name} space {len(tsk.config_space)}")

        tuner_obj = OneShotTuner(tsk, plan_size=opt.n_parallel, optimizer=opt.optimizer, opt=opt)

        tsk_trial = min(n_trial, len(tsk.config_space))
        callback = [
            autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
            autotvm.callback.log_to_file(f"{save_dir}/{model_name}.log"),
        ]
        tuner_obj.one_shot_tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=callback,
            opt=opt,
        )
        end = time.time()
        print(f"\nindex {i+1} time {end-start}")
    autotvm.record.pick_best(log_file, best_log_file)


def tune_and_evaluate(tuning_opt):
    import tvm
    import tvm.contrib.graph_runtime as runtime
    from tvm import autotvm
    from tvm import relay
    import utils

    # extract workloads from relay program
    try:
        print("Extract tasks...")
        mod, params, input_shape, out_shape = utils.get_network(
            model_name, batch_size=batch_size, layout=opt.layout
        )
        if model_name == "bert":
            tasks = autotvm.task.extract_from_program(
                mod["main"],
                target=target,
                params=params,
            )
        else:
            tasks = autotvm.task.extract_from_program(
                mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
            )
        # run tuning tasks
        print("Tuning...")
        start = time.time()
        tune_tasks(tasks, **tuning_opt)
        end = time.time()

        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "Total optimization time {:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        )
        # compile kernels with history best records
        mod, params, input_shape, out_shape = utils.get_network(
            model_name, batch_size=batch_size, layout=opt.layout
        )

        np.save(os.path.join(save_dir, "end2end.npy"), end - start)

        with autotvm.apply_history_best(best_log_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)

            # load parameters
            ctx = tvm.context(str(target), 0)
            module = runtime.GraphModule(lib["default"](ctx))
            if model_name == "bert":
                tt_a = tvm.nd.array(input_shape, ctx)
                st_a = tvm.nd.array(out_shape, ctx)
                module.set_input("input_ids", tt_a)
                module.set_input("attention_mask", st_a)
                module.set_input(**params)
            else:
                data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
                module.set_input("data", data_tvm)

            # evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print(
                "Mean inference time (std dev): %.2f ms (%.2f ms)"
                % (np.mean(prof_res), np.std(prof_res))
            )
            np.save(os.path.join(save_dir, "second.npy"), prof_res)

    except Exception as e:
        import traceback
        import sys

        traceback.print_exc()
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        sys.exit("fatal error")


if __name__ == "__main__":
    import tvm
    from tvm import autotvm

    save_dir = f"{os.path.dirname(os.path.realpath(__file__))}"
    print(f"option", opt)
    save_dir = os.path.join(
        save_dir, opt.save, opt.network, opt.layout, str(opt.batch), f"{opt.optimizer}"
    )
    opt.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creating folder {format(save_dir)}")
    dtype = "float32"
    batch_size = opt.batch
    model_name = opt.network
    log_file = f"{save_dir}/{model_name}.log"
    best_log_file = f"{save_dir}/best_{model_name}.log"
    num_threads = opt.thread  # 1
    os.environ["TVM_NUM_THREADS"] = str(opt.thread)

    target = tvm.target.Target(opt.target)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(opt), f, indent=2)
    logging.info(
        "Saving configuration file in `{0}`".format(
            os.path.abspath(os.path.join(save_dir, "config.json"))
        )
    )
    tuning_option = {
        "log_filename": log_file,
        "tuner": opt.optimizer,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10, n_parallel=opt.n_parallel),
            runner=autotvm.LocalRunner(number=8, repeat=2, timeout=20),
        ),
        "n_trial": opt.n_trial,
    }
    tune_and_evaluate(tuning_option)
