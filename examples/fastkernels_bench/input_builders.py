"""Generate valid operator inputs while preserving the captured tensor shapes."""


def copy_strided(destination, source):
    slices = tuple(slice(0, 1) if stride == 0 else slice(None) for stride in destination.stride())
    destination[slices].copy_(source[slices])


def normalize_init(call):
    values = {k: v for k, v in call.items() if isinstance(v, (int, float, bool, str, list))}
    config = call.get("config", {})
    if isinstance(config, dict) and "$dataclass" in config:
        values.update(config.get("fields", {}))
    return values


def build(op, fwd_args, device, init_args, materialize):
    import torch

    values = materialize(fwd_args, device, init_args)
    if op.endswith(":OneHot"):
        classes = int(values["num_classes"])
        if classes == 0 or classes < -1:
            raise ValueError("One-hot encoding requires positive classes or automatic inference.")
        if classes > 0:
            values["x"] = values["x"].remainder(classes)
    if op.endswith(":EmbeddingBag"):
        ids, offsets = values["input_ids"], values["offsets"]
        bags = offsets.numel() - int(init_args.get("include_last_offset", False))
        if bags <= 0:
            raise ValueError("Embedding bags require a positive bag count.")
        values["offsets"] = torch.arange(offsets.numel(), device=device, dtype=offsets.dtype) * ids.numel() // bags
    if "sparse_indices" in values:
        sizes = init_args["num_embeddings_per_feature"]
        if len(values["sparse_indices"]) != len(sizes):
            raise ValueError("Feature count does not match the captured vocabulary sizes.")
        values["sparse_indices"] = [x.remainder(size) for x, size in zip(values["sparse_indices"], sizes, strict=True)]
    for name, size_name in (("user_ids", "num_users"), ("item_ids", "num_items")):
        if name in values:
            values[name] = values[name].remainder(init_args[size_name])
    return [], values
