#!/usr/bin/env python
import click
import numpy as np
import torch
import triton
from operator_inp_utils import OperatorInputsLoader

import torchinductor
from torchdynamo.optimizations.backends import cudagraphs_inner
from torchdynamo.testing import same
from torchinductor.compile_fx import compile_fx
from torchinductor.utils import gen_gm_and_inputs
from torchinductor import config as inductor_config

from torchinductor.lowering import lowerings, fallbacks
from torchinductor.decomposition import decompositions
from torch import tensor

aten = torch.ops.aten


def compute_speedups(repeats, models, example_inputs):
    expected = models[0](*example_inputs)
    # for model in models[1:]:
    #     actual = model(*example_inputs)
        # assert same(actual, expected), expected[0] - actual[0]

    timings = np.zeros((repeats, len(models)), np.float64)
    for rep in range(repeats):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            # do_bench() clears L2 cache to hide the latency of CPU launch time
            # along with cuda synchronization
            median_ms, _, _ = triton.testing.do_bench(lambda: model(*example_inputs))
            timings[rep, m] = median_ms
    return np.median(timings, axis=0)


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

def convert_to_jit(gm, gm_args):
    strip_overloads(gm)
    try:
        return torch.jit.script(gm)
    except Exception :
        pass
    return torch.jit.trace(gm, gm_args)

def microbenchmark(target, args, kwargs, dtype):
    gm, gm_args = gen_gm_and_inputs(target, args, kwargs)
    torch.jit._builtins._register_builtin(torch.ops.aten.convolution_backward.default, "aten::convolution_backward")
    compiled_fn = compile_fx(gm, gm_args)
    cudagraphs_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False)
    g = convert_to_jit(gm, gm_args)
    cudagraphs_jit = cudagraphs_inner(
        g, gm_args, copy_outputs=False
    )

    repeats = 3
    medians = compute_speedups(
        repeats,
        [cudagraphs_eager, cudagraphs_jit, compiled_fn],
        gm_args,
    )
    return medians
    # print(f"Perf for {target} {dtype} w/cudagraphs")
    # print(f"JIT NVFuser speedup over aten {medians[0]/medians[1]}")
    # print(f"Inductor speedup over aten {medians[1]/medians[2]}")

def skip_operator(operator):
    if "nll_loss" in str(operator):
        # TODO - inputs cannot be randomly initialized, causes cyda failures
        print(f"Skipping {operator}, input generator nyi")
        return True

    # not covered by other non-compute operator heuristics
    if operator == torch.ops.aten._unsafe_view.default:
        print(f"Skipping {operator}, non compute operator")
        return True

    op_impls = [operator]
    if isinstance(operator, torch._ops.OpOverload):
        op_impls.append(operator.overloadpacket)

    if all(op not in decompositions for op in op_impls) and any(op in fallbacks for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if "convolution" in str(operator) and inductor_config.triton.convolution == "aten":
        return True

    if inductor_config.triton.mm == "aten" and operator == aten.mm.default:
        return True

    return False
    
@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark")
@click.option("--dtype", help="dtype to benchmark")
def benchmark(suite, op, dtype):
    assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
    if suite == "timm":
        loader = OperatorInputsLoader.get_timm_loader()
    elif suite == "huggingface":
        loader = OperatorInputsLoader.get_huggingface_loader()
    else:
        loader = OperatorInputsLoader.get_torchbench_loader()

    assert dtype in ("float16", "float32"), f"got {dtype}"
    dtype = torch.float16 if dtype == "float16" else torch.float32

    f = open(f"timings_{suite}_{dtype}.txt", "a")

    for operator in loader.get_all_ops():
        if skip_operator(operator):
            continue

        print(f"Running {operator}")
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype)
        timings = []
        while True:
            try:
                args, kwargs = next(inp_gen)
            except StopIteration:
                break

            try:
                out = microbenchmark(operator, args, kwargs, dtype)
                # aten, nvfuser, inductor
                timings.append(microbenchmark(operator, args, kwargs, dtype))
            except Exception as e:
                print(f"error {operator}")
                print(e)
                pass
        timings = torch.tensor(timings).T
        q = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        output = f"\n{operator}:\nNVFUSER Speedups : {(torch.quantile(timings[0] / timings[1], q)).tolist()}"
        output = f"{output}\nInductor Speedups : {(torch.quantile(timings[0] / timings[2], q)).tolist()}"
        f.write(output)
        print(output)

    f.close()


if __name__ == "__main__":
    # TODO: getting error w/aot_autograd in compile fx
    torchinductor.config.aot_autograd = False
    benchmark()
