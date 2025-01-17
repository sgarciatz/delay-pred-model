# Monte Carlo Sampling
This repository holds the code implementation of a Delay Prediction Model based on a Monte Carlo Sampling process. 

The implementation is seeded so that results shown in the paper "DP-DT: Data Plane Digital Twin Architecture to Handle Conflicts among SDN Applications" are reproducible.

# How to run it?

First, clone the repo

    git clone https://github.com/sgarciatz/delay-pred-model.git

Then, create and start a virtual environment inside the directory:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Finally execute the "monte_carlo_sampling.py" script.

    python monte_carlo_sampling.py


