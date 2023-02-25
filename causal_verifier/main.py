import torch as t
import torch.nn as nn


# from jaxtyping import Array
# from typeguard import typechecked

# patch_typeguard()  # use before @typechecked
# @typechecked

class Predicate:
    def __call__(self, data):
        return 0


class Hook:
    def __init__(self, layer: t.nn.Module, pred: Predicate):
        self.handle = layer.register_forward_hook(self)
        self.inp = None
        self.out = None
        self.pred = pred

    def __call__(self, layer, inp, out):
        self.inp = inp[0]
        self.out = out
        with t.no_grad():
            self.pred_res = self.pred(self.inp)
            replacement_lookup = self.pred_res[..., None, :] == self.pred_res[..., :, None]
            replacement_lookup.diagonal()[replacement_lookup.sum(dim=0) != 1] = False
            replacement_lookup = replacement_lookup.to(dtype=float)
            replacement_lookup = replacement_lookup / replacement_lookup.sum()
            replace_indices = t.multinomial(replacement_lookup, 1)[:, 0]
        return out[replace_indices.detach()]

    def clear(self):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


class HookedModel(nn.Module):
    def __init__(self, model: t.nn.Module, mappings: dict, hook_class=Hook):
        """
        HookedModel - a model with hooks to resample the activations
        :param model: the unmodified model we'd like to verify our hypothesis for
        :param mappings: a dictionary of layer -> predicate mappings, our hypothesis
        :param hook_class: optional, to specify your own hook
        """
        super().__init__()
        self.model = model
        self.mappings = mappings
        self.hooks = [hook_class(layer, pred) for layer, pred in mappings.items()]

    def forward(self, x):
        return self.model(x)
