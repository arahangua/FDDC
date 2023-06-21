# Fractional dynamic differential covariance (FDDC)
Repo for the paper "Scalable covariance based connectivity inference for synchronous neuronal networks".

Biorxiv preprint: [https://www.biorxiv.org/content/10.1101/2023.06.17.545399v1](https://www.biorxiv.org/content/10.1101/2023.06.17.545399v1)

# Requirements 

```
pip install -r requirements.txt
```


# Description
## Folder structure 
```
├── coninf 
│   ├── coninf.ipynb  # python notebook file for running multivariate inference methods
│   ├── matlab  # all matlab scripts for running pairwise inference (CCG) 
│   └── python  # all python scripts for the multivariate inference methods
├── example_data  # default output location of simulations
│   └── 2000 
├── LICENSE
├── README.md
├── requirements.txt
├── simulation
│   ├── adex_model_script_noise.py # the script for running noise-perturbed simulations
│   └── adex_model_script.py # main script for running simulations
└── util
    ├── __pycache__
    └── util.py  # utility functions 
```


# Usage

## Running LIF simulations 
example command: 

```
python -u adex_model_script.py --sim_time 1200 --N 2000 --a 13 --b 24 --instance 0  
```
* sim_time : simulation length in seconds (s)
* N : number of neurons to be simulated (at the moment, excitatory - inhibitory ratio is fixed. 80% vs 20%)
* a : adaptation parameter (nS)
* b : adaptation parameter (pA)
* instance : integer alias of the simulation in case we want to replicate multiple simulations with the same parameter set. 

- **make sure to set 'root_dir' variable in the script correctly before running simulations.**


For simulations with noise 
```
python -u adex_model_script_noise.py --sim_time --1200 --N 2000  --a 13 --b 24 --instance 0 --noise 0.05 
```
* noise : noise in nA 

- make sure to run simulations w/o noise first (please set the same 'root_dir' for both simulation scripts), as the noised simulation uses the same connections and synaptic delays of the non-noised simulation. 


## Inferring connectivity 
Please check the notebook file ("coninf/coninf.ipynb") and the readme file inside the 'coninf/matlab' folder. 
