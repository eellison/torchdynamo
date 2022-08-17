
#!/usr/bin/env python
import argparse
import collections
import copy
import csv
import functools
import gc
import io
import itertools
import logging
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from microbenchmarks.operator_inp_utils import OperatorInputsMode, deserialize_tensor, OperatorInputsLoader
from scipy.stats import gmean
from scipy.stats import ttest_ind
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map

import torchdynamo
import torchdynamo.utils
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.inference import fixed_strategy1
from torchdynamo.optimizations.inference import fixed_strategy2
from torchdynamo.optimizations.inference import offline_autotuner
from torchdynamo.optimizations.inference import online_autotuner
from torchdynamo.optimizations.log_args import conv_args_analysis
from torchdynamo.optimizations.python_key import python_key
from torchdynamo.profiler import Profiler
from torchdynamo.profiler import fx_insert_profiling
from torchdynamo.testing import dummy_fx_compile
from torchdynamo.testing import format_speedup
from torchdynamo.testing import same

import torchvision 


model = torchvision.models.resnet18(pretrained=True)
example_inputs = torch.rand([4, 3, 255, 255])

operator_mode = OperatorInputsMode()
fake_tensor_mode = FakeTensorMode()

with torch._subclasses.fake_tensor.FakeCopyMode(fake_tensor_mode):
    model_fake = copy.deepcopy(model)
    example_inputs_fake = copy.deepcopy(example_inputs)
with fake_tensor_mode, operator_mode:
    out = model_fake(example_inputs_fake)


# for val in operator_mode.func_db["aten.convolution.default"].items():
#     import pdb; pdb.set_trace()
#     out = eval(val[0], {"T": deserialize_tensor})
#     print(val)
# import pdb; pdb.set_trace()
operator_mode.log_to_file("/scratch/eellison/work/elias.txt")
# loader = OperatorInputsLoader("/scratch/eellison/work/torchdynamo/op_outputs/tmp/dlrm_training.json")
# inps = loader.get_inputs_for_operator("aten.cat.default", dtype=torch.float)
# print(next(inps))
# print(out)


