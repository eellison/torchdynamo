import json
import logging
import os
import zipfile
from collections import Counter
from collections import defaultdict
from functools import partial
from os.path import exists
from typing import Any
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Tuple

import functools
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

OP_INP_DIRECTORY = os.path.join(os.path.dirname(__file__), "operator_inp_logs")

TIMM_FILE = os.path.join(OP_INP_DIRECTORY, "timm_train_inps")
HF_FILE = os.path.join(OP_INP_DIRECTORY, "hf_train_inps")
TORCHBENCH_FILE = os.path.join(OP_INP_DIRECTORY, "torchbench_train_inps")

aten = torch.ops.aten

def deserialize_tensor(size, dtype, stride=None):
    size = args[0]
    ind = 1
    stride = None
    if type(args[1]) is not str:
        stride = args[1]
        ind += 1
    dtype = eval(f"torch.{args[ind]}")
    if stride:
        return torch.empty_strided(size=size, stride=stride, dtype=dtype)
    else:
        return torch.empty(size=size, dtype=dtype)


class ReprHolder:
    def __init__(self, call, args, kwargs=None):
        self.call = call
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
    
    def __repr__(self):
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = ", ".join([f"{key}={value}" for key, value in self.kwargs.items()])
        return f"{self.call}({args}{kwargs})"


def deserialize_sparse_tensor(size, dtype, layout, is_coalesced, nnz=None):
    assert layout == torch.sparse_coo
    args = [size, dtype, layout, is_coalesced]
    if nnz:
        args.append(nnz)
    return ReprHolder("ST", args)

def serialize_sparse_tensor(e):
    if isinstance(e, torch._subclasses.FakeTensor):
        return ReprHolder("ST", (list(e.shape), e.dtype, e.layout, e.is_coalesced()))
    else:
        return ReprHolder("ST", (list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()))

def deserialize_tensor(size, dtype, stride=None):
    if stride is not None:
        return torch.empty_strided(size, stride, dtype=dtype)
    else:
        return torch.empty(size, dtype=dtype)

def serialize_tensor(e):
    if not e.is_contiguous():
        return ReprHolder("T", (list(e.shape), e.dtype, e.stride()))
    else:
        return ReprHolder("T", (list(e.shape), e.dtype))

def serialize_torch_args(e):
    if isinstance(e, torch.Tensor):
        if e.is_sparse:
            return serialize_sparse_tensor(e)
        return serialize_tensor(e)
    elif isinstance(e, torch.device):
        return f"torch.device('{e.type}')"
    else:
        return e


def contains_tensor(elems):
    for elem in tree_flatten(elems)[0]:
        if isinstance(elem, torch.Tensor):
            return True
    return False


def skip_args(elems):
    for i in tree_flatten(elems)[0]:
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False

tensor_type = torch._C.TensorType.get()

def contains_tensor_types(type):
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )

@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema

    # skip constructors
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True
    if "_like" in op.name:
        return True

    # allow in place writes
    if schema.is_mutable:
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]
    
    # skip aliasing unless there are multiple outputs
    if len(tensor_outputs) != 1:
        return False

    schema_info = torch._C._SchemaInfo(schema)
    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(tensor_outputs[0].alias_info.after_set):
                return True

    return False

class OperatorInputsMode(TorchDispatchMode):
    def __init__(self, func_db=None):
        self.func_db = defaultdict(Counter) if func_db is None else func_db

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        arg_meta, kwarg_meta = tree_map(serialize_torch_args, (args, kwargs))

        out = func_overload(*args, **kwargs)

        inps = (args, kwargs)
        if contains_tensor(inps) and not skip_args(inps) and contains_tensor(out):
            serialized_str = repr((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][serialized_str] += 1

        return out

    def log_to_file(self, output_filename, *, non_compute_operators=False):
        sorted_operators = sorted(list(self.func_db.keys()))
        with open(output_filename, "w") as f:
            for operator in sorted_operators:
                if non_compute_operator(eval(operator)):
                    continue
                f.write(f"Operator: {operator}\n")
                operator_inputs = self.func_db[operator]
                for inps, count in operator_inputs.items():
                    inp_repr = repr(inps).replace("torch", "th")
                    f.write(f"cnt: {count}, {inp_repr}\n")

def map_to_device(e, device):
    return e.to(device) if isinstance(e, torch.Tensor) else e


def map_to_dtype(e, dtype):
    if isinstance(e, torch.Tensor) and e.is_floating_point():
        return e.to(dtype)
    else:
        return e

def eval_inp_str(inps):
    return eval(inps.strip().strip("'"), {"T": deserialize_tensor, "ST": deserialize_sparse_tensor, "th": torch})

class OperatorInputsLoader:
    def __init__(self, json_file_path):
        self.operator_db = {} 
        with open(json_file_path, "r") as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            op_line = lines[i].strip("\n")
            assert "Operator: " in op_line, op_line
            operator = op_line[len("Operator: "):]
            op_inps = Counter()
            i += 1
            while i < len(lines) and "Operator: " not in lines[i]:
                line = lines[i]
                cnt = eval(line[len("cnt: "):line.find(",")])
                inps = line[line.find(",") + 2:].strip("'")
                op_inps[inps] = cnt
                i += 1
            self.operator_db[operator] = op_inps
            

    def get_inputs_for_operator(
        self, operator, dtype, device="cuda"
    ) -> Generator[Tuple[Iterable[Any], Dict[str, Any]], None, None]:
        assert (
            str(operator) in self.operator_db), f"Could not find {operator}, must provide overload"

        if "embedding" in str(operator):
            logging.warn("Embedding inputs NYI, input data cannot be randomized")
            yield
            return

        # counter represents number of times these inputs occured, ignored for now
        for inps, counter in self.operator_db[str(operator)].items():
            args, kwargs = eval_inp_str(inps)

            to_dtype = partial(map_to_dtype, dtype=dtype)
            args, kwargs = tree_map(to_dtype, (args, kwargs))

            if device:
                to_device = partial(map_to_device, device=torch.device(device))
                args, kwargs = tree_map(to_device, (args, kwargs))

            yield args, kwargs

    def get_all_ops(self):
        for key in self.operator_db.keys():
            yield eval(key)

    def get_call_frequency(self, op):
        assert (
            str(op) in self.operator_db
        ), f"Could not find {op}, must provide overload"

        count = 0
        for _, counter in self.operator_db[str(op)].items():
            count += counter
        return count

    @staticmethod
    def get_timm_loader():
        return OperatorInputsLoader._get_loader(TIMM_FILE)

    @staticmethod
    def get_huggingface_loader():
        return OperatorInputsLoader._get_loader(HF_FILE)

    @staticmethod
    def get_torchbench_loader():
        return OperatorInputsLoader._get_loader(TORCHBENCH_FILE)

    @staticmethod
    def _get_loader(inp_path):
        json_path = inp_path + ".json"
        if not exists(json_path):
            zip_path = inp_path + ".zip"
            assert exists(zip_path), f"Could not find {inp_path}"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(OP_INP_DIRECTORY)

        return OperatorInputsLoader(json_path)
