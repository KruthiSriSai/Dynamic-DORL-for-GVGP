# Dynamic Dual Observation Reinforcement Learning for General Video Game Playing (D-DORL)

## Project Overview

This project implements a Dynamic Dual Observation Reinforcement Learning (D-DORL) framework for General Video Game Playing (GVGP).

The key idea is to allow a reinforcement learning agent to learn using both:

1) Global observation of the environment  
2) Local multi-scale observations around the agent  

These observations are fused using an attention-based neural network, enabling the agent to dynamically focus on the most relevant information while making decisions.

The framework is evaluated across multiple game environments to demonstrate generalization and robustness.

---

## Objectives

- Design a dual-observation RL framework  
- Implement multi-scale local observation extraction  
- Integrate attention-based feature fusion  
- Train agents using Proximal Policy Optimization (PPO)  
- Evaluate performance on different video game environments  

---

## Technologies and Tools

- Python  
- PyTorch  
- Stable-Baselines3  
- OpenAI Gym / Gymnasium  
- NumPy  

---

## Project Structure

```
dl_demo/
│
├── dlo_demo.py
├── lunarlander_dlo.py
├── hillclimb_dlo.py
├── jetpack_dlo.py
├── subway_dlo.py
│
├── requirements.txt
│
├── ppo_dlo_demo.zip
├── ppo_lunar_dlo.zip
├── ppo_hill_dlo.zip
├── ppo_jetpack_dlo.zip
├── ppo_subway_dlo.zip
│
├── README.md
└── LICENSE
```


---

## Installation

1. Clone the repository

   git clone https://github.com/KruthiSriSai/Dynamic-DORL-for-GVGP.git

   cd Dynamic-DORL-for-GVGP

2. Create virtual environment (recommended)

   python -m venv venv

   venv\Scripts\activate

3. Install dependencies

   pip install -r requirements.txt

---

## Running the Project

### Train an agent

   python dlo_demo.py train

### Evaluate an agent

   python dlo_demo.py eval

### For other environments:

   python lunarlander_dlo.py train  
   python lunarlander_dlo.py eval  

   python hillclimb_dlo.py train  
   python hillclimb_dlo.py eval  

   python jetpack_dlo.py train  
   python jetpack_dlo.py eval  

   python subway_dlo.py train  
   python subway_dlo.py eval  

---

## Results

The D-DORL framework shows improved learning stability and performance compared to single-observation baselines.  
Pre-trained PPO models are provided in the repository.

---

## Applications

- Game AI  
- Robotics and control systems  
- Autonomous agents  
- Research in reinforcement learning  

---

## Future Work

- Add more GVGP environments  
- Hyperparameter tuning  
- Compare with additional RL algorithms  
- Visualization of attention weights  

---

## Authors

Kruthi Sri Sai
Umesh Yenduru
Undergraduate Major Project  

---

## License

MIT License
