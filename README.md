# Experiments for the RATS algorithm

This repository contains an implementation of the RATS
algorithm presented in the paper "Non-Stationary Markov
Decision Processes a Worst-Case Approach using Model-Based
Reinforcement Learning" along with the reported experiments.


## Installation and use

First clone the repository:

```bash
git clone git@github.com:SuReLI/rats-experiments.git
cd rats-paper-experiments
```

Ensure all the dependencies described in the next section are
installed within your python environment. 
Now you can do your stuff.

```bash
python example.py
```

## List of dependencies

Standard dependencies include the following:
`numpy`,
`csv`,
`multiprocessing`,
`time`,
`itertools`,
`scipy`,
`math`.

Non-standard dependencies include `gym`,
available [at this location](https://github.com/openai/gym).

## Run the experiments of the paper

To reproduce the experiment of the paper "Non-Stationary
Markov Decision Processes a Worst-Case Approach using
Model-Based Reinforcement Learning", run the following
script sequentially:

	python nsbridge_experiment.py
	python results_exploitation.py

## Quote

If you use this code in your work or build on the
[paper](https://arxiv.org/abs/1904.10090), please cite the
latter.
Here is an example of bibtex entry:

    @inproceedings{lecarpentier2019non,
        title={{Non-Stationary Markov Decision Processes a Worst-Case Approach using Model-Based Reinforcement Learning}},
        author={Lecarpentier, Erwan and Rachelson, Emmanuel},
        booktitle={Advances in Neural Information Processing Systems}
    }
