import gc

import torch
from torch.autograd import Function
from torch.nn import Module
from torch.nn.parameter import Parameter

import operator


def jacobian(f, x, eps):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = torch.eye(len(x)).type_as(get_data_maybe(x))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps * e[i]) - f(x - eps * e[i])) / (2. * eps))
    J = torch.stack(J).transpose(0, 1)
    return J


def expandParam(X, n_batch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([n_batch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def bdiag(d):
    assert d.ndimension() == 2
    nBatch, sz = d.size()
    dtype = d.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.view(-1)
    return D


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def eclamp(x: torch.Tensor, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x


def get_data_maybe(x):
    return x


_seen_tables = []


def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print('| ' + ' | '.join(r) + ' |')

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2, 3]
        if len(di) == 3:
            e, fmt = di[1:]
            s.append(fmt.format(e))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_traj(T, u, x_init, dynamics, debug_memory_mode=False):
    from .mpc import QuadCost, LinDx  # TODO: This is messy.
    with torch.no_grad():
        if isinstance(dynamics, LinDx):
            F = get_data_maybe(dynamics.F)
            f = get_data_maybe(dynamics.f)
            if f is not None:
                assert f.shape == F.shape[:3]
            if debug_memory_mode:
                F.debug_name = f"(GT) F \t"
                f.debug_name = f"(GT) f \t"
        x = torch.zeros(T, x_init.shape[0], dynamics.n_state, device=x_init.device)
        if debug_memory_mode:
            x.debug_name = f"(GT) x_traj"
        x[0] = get_data_maybe(x_init)
        for t in range(T - 1):
            xt = x[t]
            ut = u[t]
            if isinstance(dynamics, LinDx):
                xut = torch.cat((xt, ut), 1)
                new_x = bmv(F[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                new_x = dynamics(xt, ut).data
            x[t + 1] = new_x
    del xt, ut, new_x  # successfully deleted
    return x


last_tensor_count = 0
last_state = "NoState"


def print_torch_memory_allocated(state: str = None, print_tensors: bool = False, delete_before: bool = False):
    global last_tensor_count, last_state
    if delete_before:
        torch.cuda.empty_cache()
        gc.collect()

    if state is not None:
        print(last_state + " -> " + state)
        last_state = state
    tensor_count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if print_tensors and False:
                    print("\tvariable_name: " + ("NoSetName" if not hasattr(obj, "debug_name") else obj.debug_name),
                          "\t", obj.size(),
                          )
                tensor_count += 1
                if obj.requires_grad:
                    debug = 0

        except:
            pass
    print("Tensor count: " + str(tensor_count), "Delta: " + str(tensor_count - last_tensor_count))
    last_tensor_count = tensor_count


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    from .mpc import QuadCost, LinDx  # TODO: This is messy.

    assert x_init is not None or x is not None

    if isinstance(cost, QuadCost):
        C = get_data_maybe(cost.C)
        c = get_data_maybe(cost.c)

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        if isinstance(cost, QuadCost):
            obj = 0.5 * bquad(xut, C[t]) + bdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj


def detach_maybe(x, debug_memory_mode=True):
    if x is None:
        return None
    if x.requires_grad:
        dx = x.detach()
        if debug_memory_mode:
            dx.debug_name = f"(Detached) {x.debug_name if hasattr(x, 'debug_name') else 'NoSetName'}"
        return dx
    return x


def data_maybe(x):
    if x is None:
        return None
    return x.data
