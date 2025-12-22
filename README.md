
**Transformer FlOW**

1. project_feature_eng_model.py --> for feature generation and pickle data set

2. project_transformer_training.py --> for training the transformer based on the pickle file from previous step.

3. project_compressed_generator_transformer.py --> produces a compressed file that stacks together the forecasts of the transformer + the 12 features, the output of this script is fed into the Deep RL

4. project_rl.py --> Trains the Deep RL agents with the compressed generated file

5. project_rl_inference_trans.py --> Is the file for generating the performance of the deep RL in new unseen data, creates performance plots, uses backtesting script inside, gives equity curve, trades taken and training logs.

---

**LightGBM FLOW**

1. project_feature_eng_model.py --> for feature generation and pickle data set

2. project_lgb_training.py --> trains a LGB with the pickle file generated from previous step

3. project_compressed_generator_lightgbm.py --> produces a compressed file that stacks together the forecasts of the lgb + the 12 features, the output of this script is fed into the Deep RL

4.  project_rl.py --> Trains the Deep RL agents with the compressed generated file

5. project_rl_inference_lgb.py --> Is the file for generating the performance of the deep RL in new unseen data, creates performance plots, uses backtesting script inside, gives equity curve, trades taken and training logs.

---

Extra files

1. project_backtest.py --> is used by DRL, must be in DIR or it wont work

2. pickle_test.py --> is a testing file to inspect the pickle files generated from other scripts