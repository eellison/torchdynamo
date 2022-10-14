import torch
import torchdynamo
import torch.nn.functional as F

# def get_peak_memory():
#     return torch.cuda.max_memory_allocated() / 10**9

# def reset():
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.empty_cache()

# def mish_fwd(x):
#     return (x * 3) * 2

# mish_opt = torchdynamo.optimize("inductor")(mish_fwd)

# print(mish_opt(torch.ones([8000]).cuda()).sum())
# print(mish_opt(torch.ones([4000]).cuda()).sum())
# print(mish_opt(torch.ones([16000]).cuda()).sum())

# def new_fn(a, b, c):
#     return a + b + c

# new_fn_opt = torchdynamo.optimize("inductor")(new_fn)
# print("RUNNING AGAIN")
# new_fn_opt(torch.ones([4000]).cuda(), torch.ones([4000]).cuda(), torch.ones([4, 4000]).cuda())

def foo(x, y):
    return x[y] + 1 + 2

foo_opt = torchdynamo.optimize("inductor")(foo)

x = torch.rand([100, 100]).cuda()
y = torch.ones([100], dtype=torch.long).cuda()

out1 = foo(x, y)
out2 = foo_opt(x, y)

torch.allclose(out1, out2)
# print(out1, out2)



# for fn in (mish_opt,):
#     reset()
#     if fn is mish_fwd:
#         print("EAGER")
#     else:
#         print("IND")
    
#     inp = torch.rand([100000]).cuda().requires_grad_(True)
#     out = mish_opt(inp)
#     print("fwd mem", get_peak_memory())
#     out.sum().backward()
#     print("backward mem", get_peak_memory())
#     del inp
