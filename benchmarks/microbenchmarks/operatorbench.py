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
    # compiled_fn = compile_fx(gm, gm_args)
    # cudagraphs_eager = cudagraphs_inner(gm, gm_args, copy_outputs=False)
     
    g = convert_to_jit(gm, gm_args)
    cudagraphs_jit = cudagraphs_inner(
        g, gm_args, copy_outputs=False
    )

    # try:
    #     g = torch.jit.trace(gm, gm_args)
    # except Exception:
    #     try:
    #         g = torch.jit.script(gm.code)
    #     except Exception
    #         g = None

    # except Exception as e:
    #     import pdb; pdb.set_trace()
    #     print(e)

    # repeats = 3
    # medians = compute_speedups(
    #     repeats,
    #     [cudagraphs_eager, cudagraphs_jit, compiled_fn],
    #     gm_args,
    # )

    # print(f"Perf for {target} {dtype} w/cudagraphs")
    # print(f"JIT NVFuser speedup over aten {medians[0]/medians[1]}")
    # print(f"Inductor speedup over aten {medians[1]/medians[2]}")


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

    for operator in loader.get_all_ops():
        if "convolution_backward" not in repr(operator):
            continue
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype)
        while True:
            try:
                inps = next(inp_gen)
            except Exception as e:
                print(operator)
                print(e)
                break
            args, kwargs = next(inp_gen)
            microbenchmark(operator, args, kwargs, dtype)
            break


if __name__ == "__main__":
    # TODO: getting error w/aot_autograd in compile fx
    torchinductor.config.aot_autograd = False
    benchmark()
