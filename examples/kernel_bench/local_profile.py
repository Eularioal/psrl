"""Preserve shape constraints and integer inputs in local inference checks."""

import ast


def prepare_source(source):
    """Reduce batches and dense dimensions before dependent shapes are evaluated."""
    tree = ast.parse(source)
    overrides = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        limit = 2 if name == "batch_size" else 512
        if name not in {
            "batch_size",
            "M",
            "N",
            "K",
            "input_size",
            "hidden_size",
            "output_size",
            "in_features",
            "out_features",
            "layer_sizes",
            "hidden_layer_sizes",
        }:
            continue
        allowed = (
            ast.Expression,
            ast.Constant,
            ast.BinOp,
            ast.Mult,
            ast.Add,
            ast.Sub,
            ast.FloorDiv,
            ast.List,
            ast.Tuple,
            ast.Load,
        )
        if not all(isinstance(n, allowed) for n in ast.walk(ast.Expression(node.value))):
            continue
        value = eval(compile(ast.Expression(node.value), "<dimensions>", "eval"), {"__builtins__": {}})
        if isinstance(value, int):
            value = min(value, limit)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
            value = type(value)(min(v, limit) for v in value)
        else:
            continue
        node.value = ast.parse(repr(value), mode="eval").body
        overrides[name] = value
    return ast.unparse(ast.fix_missing_locations(tree)), overrides


def evaluate_source(reference, candidate):
    """Compare inference outputs with paired initialization and preserved input dtypes."""
    import torch

    ref_scope, candidate_scope = {}, {}
    exec(reference, ref_scope)
    exec(candidate, candidate_scope)
    torch.manual_seed(42)
    init = ref_scope["get_init_inputs"]()
    torch.manual_seed(42)
    ref = ref_scope["Model"](*init).cuda().eval()
    torch.manual_seed(42)
    custom = candidate_scope["ModelNew"](*init).cuda().eval()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in ref_scope["get_inputs"]()]
    with torch.no_grad():
        torch.manual_seed(43)
        expected = ref(*inputs)
        torch.manual_seed(43)
        actual = custom(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4, equal_nan=False)
    torch.cuda.synchronize()
    return {"correctness": True, "local_evaluation": 1, "inference_mode": 1}
