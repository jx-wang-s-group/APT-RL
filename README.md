# APT-RL
Asynchronous Parallel Training (APT) for Reinforcement Learning 

This repo contains demo code of paper [Asynchronous parallel reinforcement learning for optimizing propulsive performance in fin ray control](https://link.springer.com/article/10.1007/s00366-024-02093-w), published on Engineering with Computers
> The Fish Fin Ray Control involves an in-house solver which we cannot open-source. This repo provides code controlling the Kuramoto–Sivashinsky (KS) equation environment using APT-RL. 

![image](https://github.com/user-attachments/assets/4bfaeb89-6e92-4cd3-b01b-d09be915081c)


## Run the code:
### Requirements:
This code is built upon our RL framework [PIMBRL](https://github.com/jx-wang-s-group/PIMBRL). Please refer to the reqiurement of PIMBRL for more details.

Extra packages required by APT-RL: `setproctitle` (`pip install setproctitle`).

Once you have installed the dependencies of PIMBRL, simply include the PIMBRL/src into your PYTHONPATH:
```
export PYTHONPATH=${PYTHONPATH}:PIMBRL/src
```

### Run APT-RL:
```
python asyncKS.py NUM_ENV PATH_TO_SAVE 
```
`NUM_ENV`: number of environments in parallel

`PATH_TO_SAVE`: the path of saving the results






