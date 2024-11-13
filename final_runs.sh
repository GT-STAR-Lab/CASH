#!/bin/sh

# --------------------------------
# STOP: copy-paste ENV_KWARGS into mappo_homogeneous_rnn_mpe.yaml!
# --------------------------------
echo "\nSTOP: copy-paste ENV_KWARGS into mappo_homogeneous_rnn_mpe.yaml!\n"

# firefighting env

# DAGGER
#  - CASH
python baselines/imitation_learning/dagger.py -m +alg=dagger +env=mpe_simple_fire ++alg.AGENT_HYPERAWARE=True ++env.ENV_KWARGS.capability_aware=True ++alg.AGENT_HIDDEN_DIM=2048 ++alg.AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2,0.5 ++tag=final-dagger-fire ++SEED=76,58,14
#  - RNN aware, unaware
python baselines/imitation_learning/dagger.py -m +alg=dagger +env=mpe_simple_fire ++alg.AGENT_HYPERAWARE=False ++env.ENV_KWARGS.capability_aware=True,False ++alg.AGENT_HIDDEN_DIM=4096 ++tag=final-dagger-fire ++SEED=76,58,14

# MAPPO 
#  - CASH
python baselines/MAPPO/mappo_rnn_mpe.py -m ++AGENT_HYPERAWARE=True ++ENV_KWARGS.capability_aware=True ++FC_DIM_SIZE=32 ++GRU_HIDDEN_DIM=32 ++AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2,0.5 ++tag=final-mappo-fire
#  - RNN aware, unaware
python baselines/MAPPO/mappo_rnn_mpe.py -m ++AGENT_HYPERAWARE=False ++ENV_KWARGS.capability_aware=True,False ++FC_DIM_SIZE=128 ++GRU_HIDDEN_DIM=128 ++tag=final-mappo-fire

# QMIX
#  - CASH
python baselines/QLearning/qmix.py -m +alg=qmix_mpe +env=mpe_simple_fire ++env.ENV_KWARGS.capability_aware=True ++alg.AGENT_HYPERAWARE=True ++alg.AGENT_HIDDEN_DIM=64 ++alg.AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2,0.5 ++tag=final-qmix-fire
#  - RNN aware, unaware
python baselines/QLearning/qmix.py -m +alg=qmix_mpe +env=mpe_simple_fire ++env.ENV_KWARGS.capability_aware=True,False ++alg.AGENT_HYPERAWARE=False ++alg.AGENT_HIDDEN_DIM=128 ++tag=final-qmix-fire

# ----------------------

# --------------------------------
# STOP: copy-paste ENV_KWARGS into mappo_homogeneous_rnn_mpe.yaml!
# --------------------------------
echo "\nSTOP: copy-paste ENV_KWARGS into mappo_homogeneous_rnn_mpe.yaml!\n"

# material transport

# DAGGER
#  - CASH
python baselines/imitation_learning/dagger.py -m +alg=dagger +env=mpe_simple_transport ++alg.AGENT_HYPERAWARE=True ++env.ENV_KWARGS.capability_aware=True ++alg.AGENT_HIDDEN_DIM=2048 ++alg.AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2 ++tag=final-dagger-hmt ++SEED=76,58,14
# ln ablation
# python baselines/imitation_learning/dagger.py -m +alg=dagger +env=mpe_simple_transport ++alg.AGENT_HYPERAWARE=True ++env.ENV_KWARGS.capability_aware=True ++alg.AGENT_HIDDEN_DIM=2048 ++alg.AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2 ++alg.AGENT_HYPERNET_KWARGS.USE_LAYER_NORM=False ++tag=final-dagger-hmt-ln ++SEED=76,58,14

#  - RNN aware, unaware
python baselines/imitation_learning/dagger.py -m +alg=dagger +env=mpe_simple_transport ++alg.AGENT_HYPERAWARE=False ++env.ENV_KWARGS.capability_aware=True,False ++alg.AGENT_HIDDEN_DIM=4096 ++tag=final-dagger-hmt ++SEED=76,58,14

# MAPPO
#  - CASH
python baselines/MAPPO/mappo_rnn_mpe.py -m ++AGENT_HYPERAWARE=True ++ENV_KWARGS.capability_aware=True ++FC_DIM_SIZE=32 ++GRU_HIDDEN_DIM=32 ++AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2 ++tag=final-mappo-hmt
#  - RNN aware, unaware
python baselines/MAPPO/mappo_rnn_mpe.py -m ++AGENT_HYPERAWARE=False ++ENV_KWARGS.capability_aware=True,False ++FC_DIM_SIZE=128 ++GRU_HIDDEN_DIM=128 ++tag=final-mappo-hmt

# QMIX
#  - CASH
python baselines/QLearning/qmix.py -m +alg=qmix_mpe +env=mpe_simple_transport ++env.ENV_KWARGS.capability_aware=True ++alg.AGENT_HYPERAWARE=True ++alg.AGENT_HIDDEN_DIM=64 ++alg.AGENT_HYPERNET_KWARGS.INIT_SCALE=0.2 ++tag=final-qmix-hmt
#  - RNN aware, unaware
python baselines/QLearning/qmix.py -m +alg=qmix_mpe +env=mpe_simple_transport ++env.ENV_KWARGS.capability_aware=True,False ++alg.AGENT_HYPERAWARE=False ++alg.AGENT_HIDDEN_DIM=128 ++tag=final-qmix-hmt
