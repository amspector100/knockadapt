# Knockadapt

This package is intended to serve two purposes:
1. A packaged implementation of the group knockoffs framework introduced by Dai and Barber 2016 (https://arxiv.org/abs/1602.03589) for variable selection problems with highly correlated features
2. It also has a variety of algorithms for adaptively selecting groupings to maximize power while maintaing FDR control.

This is currently under heavy development (it's in the early stages): docs/tests to come.

## To run tests

Run ``python3 -m pytest``

## To do

### Abstratification 

- Probably the knockoff_stats page could be better abstract-ified

Knockoff stats:
1. test_knockoff_stats should be better abstractified

Thoughts:
1. Adaptive module should have a class which has
sample_split, double_dipping procedure, so they're 

### Naming conventions

- Basically follow the knockoff package in R (which is beautiful)
- Also can probably add useful pieces of functionality based on the knockoff
filter package (e.g., add a mu option for group gaussian knockoffs)

### Adaptive to-do

1. Finish test_eval_knockoff_instance and all tests below this
2. Possibly move power calculation to its own function? so we can test it separately?

## Bugs that must be dealt with
1. weird stdout flush error in experiments
2. SVD error in multivariate normal sampling - 
this is low-priority