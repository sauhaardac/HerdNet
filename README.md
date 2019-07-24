# Herding with Reinforcement Learning

Optimal Control Coupled Reinforcement Learning for Swarm Control

## Installation Instructions
1. Download the Anaconda bash Script
```bash
cd /tmp
curl -O curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
```
2. Run the Anaconda Script
```bash
bash Anaconda3-2019.03-Linux-x86_64.sh
```
3. Proceed with the installation and close + reopen the terminal.
4. Navigate to this cloned repo and type:
```bash
conda env create -f environment.yml -n safetynet # may take some time
conda activate safetynet
```
5. Install the OpenAI Gym environment:
```bash
cd gym/
pip install -e .
```

## Training Instructions
```bash
python train.py
```
