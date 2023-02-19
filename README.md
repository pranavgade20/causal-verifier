# Causal Verifier

A tool to verify interpretability hypothesis, based on [Redwood Research's](https://github.com/redwoodresearch/) recent [Causal Scrubbing](causal-scrubbing.pdf) work.
This only implements the first half of the work (ablate by resampling iff predicates match), while the second can be achieved by using the default predicate, and probably deviates from what is actually supposed to happen a bunch, so I'm calling it Causal Verification instead of Causal Scrubbing.

## Installation
```shell
pip install git+https://github.com/pranavgade20/causal-verifier.git
```

## Usage
Look at [example.py](examples/example.py) for a model that calculates the XOR of the input, and a script that uses this library to ablate the activations.
