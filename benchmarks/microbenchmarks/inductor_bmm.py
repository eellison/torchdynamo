import torch
from benchmark_helper import time_with_torch_timer
from torchdynamo.testing import same
import torchdynamo
import torchdynamo.config
import torchinductor.config as config

aten = torch.ops.aten

@torchdynamo.optimize("inductor", nopython=True)
def inductor_aten_bmm(x):
    return aten._adaptive_avg_pool2d(x, [2, 2])


if __name__ == "__main__":
    x = torch.rand([2, 2, 9, 9]).cuda()
    out = inductor_aten_bmm(x)
    out2 = aten._adaptive_avg_pool2d(x, [2, 2])
    # assert(same(out, out2))

