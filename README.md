# DQNNONMEM
Reproducing frozen lake DQN/RL example
Enviroment Code is adapted from
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py .
Note that on remote desktop the colorizatoin doesn't seem to work, maybe a problems with graphics on remote desktop. Alternative is to remove the
coloriztion and replace with making that character lower case.
In MyFrozenLakeEnv.py, line 204 (render function) replace 
        desc[row][col] = utils.colorize(desc[row][col], "blue", highlight=True)with:
with
        desc[row][col] = desc[row][col].lower() 
