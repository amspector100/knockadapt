# Knockadapt

This package is intended to serve two purposes:
1. A packaged implementation of the group knockoffs framework introduced by Dai and Barber 2016 (https://arxiv.org/abs/1602.03589) for variable selection problems with highly correlated features
2. It also has a variety of algorithms for adaptively selecting groupings to maximize power while maintaing FDR control.

This is currently under heavy development (it's in the early stages): docs/tests to come.

## To run tests

- To run all tests, run ``python3 -m pytest`` 
- To run a specific label, run ``pytest -v -m {label}``.
- To select all labels except a particular one, run ``pytest -v -m "not {label}"`` (with the quotes).
- To run a specific file, try pytest test/{file_name}.py. To run a specific test within the file, run pytest test/{file_name}.py::classname::test_method. You also don't have to specify
the exact test_method, you get the idea.
- To run a test with profiling, try ``python3 -m pytest {path} --profile``. This should generate a set of .prof files in prof/. Then you can run snakeviz filename.prof to visualize the output.
There are also more flags/options for outputs in the command line command.

## To do

### FX Knockoff Support

1. Knockoff Filter + Debiased Lasso
2. Need to think about whether we'll actually scale X

### Knockoff Construction

1. Add hierarchical clustering to ASDP group-making
2. Most importantly, think about how to integrate Metro.
Probably reach out to Wenshuo / Bates. 

### Graphs

1. DGP class? instead of returning like 6 things?

### Abstratification

1. Adaptive module should have a class which has
sample_split, double_dipping procedure, so they're 

### Naming conventions

- Basically follow the knockoff package in R (which is beautiful)
- Also can probably add useful pieces of functionality based on the knockoff
filter package (e.g., add a mu option for group gaussian knockoffs)

### Adaptive to-do

1. Finish test_eval_knockoff_instance and all tests below this
2. Possibly move power calculation to its own function? so we can test it separately?