python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005 

notify-send 'cs294 Training' 'Complete'

# python reference/solution.py --env_name LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 1 -l 2 -s 64 -b 20000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b20000_r0.005