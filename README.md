# Knockadapt

This package is intended to serve two purposes:
1. A packaged implementation in Python of the knockoffs framework
introduced by  

framework introduced by Dai and Barber 2016 (https://arxiv.org/abs/1602.03589) for variable selection problems with highly correlated features
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
- However, this isn't reallyyy recommended - cprofilev is much better.
To run cprofilev, copy and paste the test to proftest/* and then run 
``python3 -m cprofilev proftest/test_name.py``.


## To do

### Metro Computation

1. See footnote h on page 25. Maybe condition on the knockoffs, not 
Xjstar. This will make the within-knockoff correlations more similar
maybe?
2. 
3. 

### FX Knockoff Support

1. Knockoff Filter + Debiased Lasso
2. Need to think about whether we'll actually scale X

### Knockoff Construction

1. Add hierarchical clustering to ASDP group-making
2. Metro: should probably pass a UGM object to the metro 
class. It's much more convenient.

### MCV Computation

This can be waayy sped up. Currently block_diag 
uses almost as much time as the backprop for medium p.  
Create a file in the proftest file and then profile.

Solution: like for SDP, we should just have a version
which runs faster when there are no groups.
Examples:
- torch.diag() should run must faster than block_diag_+sparse
- Easier to store the blocks as a vector and then square them 
(instead of expensive dotting of p x p matrices)

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