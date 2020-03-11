# CS294-112 HW 1: Imitation Learning

## Behaviour Cloning

- `python behaviour_clone.py`, `python evaluate.py` to evaluate the trained model.
- The following result used expert rollout = 20, lr = 1e-3, training epoch = 2000

|             | E $\mu$  | B $\mu$ | E $\sigma$ | B $\sigma$ |
| :---------: | :------: | :-----: | :--------: | :--------: |
| Humanoid-v2 | 10686.17 | 7710.19 |   73.41    |  3586.43   |
|   Ant-v2    | 4698.77  |  4629   |   105.20   |    400     |

- `python behaviour_clone_lstm.py` to train, `python evaluate_lstm.py` to evaluate the trained model
- The following result used expert rollout = 20, lr = 1e-3, with 1 lstm layer with hidden size of 64, lr = 1e-3, training epoch = 2000

|             | E $\mu$  | B $\mu$ | E $\sigma$ | B $\sigma$ |
| :---------: | :------: | :-----: | :--------: | :--------: |
| Humanoid-v2 | 10686.17 | 9612.13 |   73.41    |  2499.10   |
|   Ant-v2    | 4698.77  | 4733.76 |   105.20   |   121.91   |

## DAgger

- Not implemented just yet. Will be done at a later date.

<!-- Modification:

We implemented the forward pass of the expert policy network in numpy, and you can use any deep learning framework you like to write this assignment.

------

Dependencies:

 * Python **3.5**
 * Numpy
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).



The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.



See the [HW1 PDF](./hw1_instructions.pdf) for further instructions. -->
