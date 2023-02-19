import torch as t
import torch.nn as nn

from causal_verifier import HookedModel, Predicate


# This is the model we'd like to hook - I've separated the first layer into two to make adding hooks easier,
# but you can either add additional layers that are identity padded with zeroes, or have a more complicated
# predicate that is a composite of multiple predicated
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 1)
        self.b = nn.Linear(2, 1)
        self.s1 = nn.Sigmoid()
        self.c = nn.Linear(2, 1)
        self.s2 = nn.Sigmoid()
        with t.no_grad():
            # set weights to perform OR, NAND, and AND operations
            self.a.weight[:] = 0
            self.b.weight[:] = 0
            self.a.weight[:] = 100
            self.b.weight[:] = -100
            self.a.bias[:] = -50
            self.b.bias[:] = 150
            self.c.weight[:] = 100
            self.c.bias[:] = -150

    def forward(self, x):
        act_or = self.a(x)
        act_nand = self.b(x)
        y = self.s1(t.cat([act_or, act_nand], dim=1))
        return self.s2(self.c(y))


# Predicates take in the input of the layer, and produce a numeric value that is same for datapoints with same value
# for some value we'd like to test
class ANDPredicate(Predicate):
    def __call__(self, x):
        return (x[..., 0] > 0.5) & (x[..., 1] > 0.5)


class NANDPredicate(Predicate):
    def __call__(self, x):
        return ~((x[..., 0] > 0.5) & (x[..., 1] > 0.5))


if __name__ == '__main__':
    net = XORNet()
    net.eval()

    # dataset - you can also use a dataloader or other methods of loading data you like, as long as it batches the input
    data_batch = t.tensor([[0, 0], [0, 1], [1, 0], [1, 1], ], dtype=t.float)
    labels = t.tensor([0, 1, 1, 0], dtype=t.float)

    # the original model's loss on our data
    print(f'Model loss: {nn.functional.mse_loss(net(data_batch)[:, 0], labels)}')

    # create a model with our hooks which will resample the activations if our predicates match
    # the dictionary maps each layer to one predicate that describes out intuition of what the layer does
    # because of duck-typing, any callable works, but using subclasses of Predicate is recommended
    # Also, just pass the default predicate if a layer is expected to not make significant impact on the loss
    hooked_net = HookedModel(
        net,
        {
            net.a: lambda data: (data[..., 0] > 0.5) | (data[..., 1] > 0.5),
            net.b: NANDPredicate(),  # ANDPredicate works here as well - they are same in the same places
            net.c: ANDPredicate()
        }
    )

    # a forward pass on the model - you can use the hooked_net like you would a normal model, including using
    # pytorch_lightning or any other framework to validate it on your dataset.
    # it is probably possible to use this to add a loss term to make sure your module ends up learning some property
    print(f'HookedModel loss: {nn.functional.mse_loss(hooked_net(data_batch)[:, 0], labels)}')
