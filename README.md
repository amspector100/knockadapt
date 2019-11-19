# Knockadapt

This package is intended to serve two purposes:
1. A packaged implementation of the group knockoffs framework introduced by Dai and Barber 2016 (https://arxiv.org/abs/1602.03589) for variable selection problems with highly correlated features
2. It also has a variety of algorithms for adaptively selecting groupings to maximize power while maintaing FDR control.

This is currently under heavy development (it's in the early stages): docs/tests to come.

## To do
- The adaptive functions should probably be a class
- The knockoff functions should probs be a class too
- Probably the knockoff_stats page could be better abstract-ified

Graph module:
1. Should have a class which samples data given covariance matrix, beta
(Takes in a family input, e.g. family = 'gaussian', 'binomial')
2. Should have a constructor function which constructs beta, 

## Bugs that must be dealt with
1. weird stdout flush error in experiments
2. SVD error in multivariate normal sampling - 
this is low-priority