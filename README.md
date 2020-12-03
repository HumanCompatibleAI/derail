Supporting code for the paper [DERAIL: Diagnostic Environments for Reward and Imitation Learning](https://arxiv.org/abs/2012.01365).

The environments are available at the [HumanCompatibleAI/seals](https://github.com/HumanCompatibleAI/seals/) repo. This repo contains the algorithms used, and the scripts to run and plot the experiments.

To reproduce the results:

```bash
git clone https://github.com/HumanCompatibleAI/derail
cd derail
pip install .

python -m derail.run -t 500000 -n 15 -p
python -m derail.plot -f results/last.csv
```
