## Installation
The code base has dependency on basic packages listed in [requirements.txt](./requirements.txt). It can be installed via the following command:
```
$ pip install -r requirements.txt 
```

## Usage
This code base implements `Private-GLM` (Algorithm 2 in the aforementioned paper). Other baseline algorithms include `RS-GLinUCB` ,`ECOLog`, `GLM-UCB` and `GLOC`, whose codes are taken from [generalized_linear_model](https://github.com/nirjhar-das/GLBandit_Limited_Adaptivity) code base of [Sawarni et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/0faa0019b0a8fcab8e6476bc43078e2e-Paper-Conference.pdf), [logistic_bandit](https://github.com/criteo-research/logistic_bandit/tree/master) code base of [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a/faury22a.pdf) and reimplemented with minor modifications. The references of the baseline algorithms are as follows:

- `ECOLog` from [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a/faury22a.pdf)
- `GLM-UCB` from [Filippi et al. 2010](https://papers.nips.cc/paper/2010/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf),
- `OL2M` from [Zhang et al. 2016](http://proceedings.mlr.press/v48/zhangb16.pdf),
- `GLOC` from [Jun et al. 2017](https://proceedings.neurips.cc/paper/2017/file/28dd2c7955ce926456240b2ff0100bde-Paper.pdf),
- `OFULogPlus` from [Lee et al. 2023](https://arxiv.org/abs/2310.18554)

For the table in Section 8, look at the notebook [simulations_extra_probit.ipynb](./simulations_extra_probit.ipynb). The first few cells generate the data and the last cells displays it as a table.

The Jupyter notebook [simulations.ipynb](./simulations.ipynb) implements the regret simulation experiements form the paper for the logistic and the probit reward models and gives the plots in Appendix A.


