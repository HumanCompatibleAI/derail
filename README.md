Supporting code for DERAIL: Diagnostic Environments for Reward and Imitation Learning

To reproduce results:

```bash
git clone https://github.com/HumanCompatibleAI/derail
cd derail
pip install .

python -m derail.run -t 500000 -n 15 -p
python -m derail.plot -f results/last.csv
```
