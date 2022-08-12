import torch
from benchmark_helper import time_with_torch_timer

import torchdynamo
import torchdynamo.config
import torchinductor.config as config
aten = torch.ops.aten

@torchdynamo.optimize("inductor", nopython=True)
def inductor_avg_pool(x):
    return (aten._adaptive_avg_pool2d(x, (1, 1)),)
    # , aten._adaptive_avg_pool2d(
    #             x + 1, (4, 5)
    #         )

# def test_total_time(shapes):
#     print("shape; torch bmm; inductor aten bmm; inductor triton bmm")
#     for i in range(len(shapes)):
#         a_shape, b_shape = shapes[i]
#         print(a_shape, "x", b_shape, end="; ")
#         a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
#         b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

#         config.triton.use_bmm = False
#         inductor_aten_bmm(a, b)

#         config.triton.use_bmm = True
#         inductor_triton_bmm(a, b)

#         torch_ms = time_with_torch_timer(torch_bmm, (a, b)).mean * 1000

#         config.triton.use_bmm = False
#         ind_aten_ms = time_with_torch_timer(inductor_aten_bmm, (a, b)).mean * 1000

#         config.triton.use_bmm = True
#         ind_triton_ms = time_with_torch_timer(inductor_triton_bmm, (a, b)).mean * 1000

#         print(torch_ms, ind_aten_ms, ind_triton_ms, sep="; ")


if __name__ == "__main__":
    import torchinductor
    torchinductor.config.debug = True
    inp = torch.randn(1, 1, 2, 2).cuda()
    out = inductor_avg_pool(inp)
    import pdb; pdb.set_trace()
    print(out)
