import os
import sys
from torchvision import transforms as tv
from PIL import Image
import numpy as np

transform = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1)])
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def display_help():

    return(
    """
    TD3 algorithm help: \n
    
    COMMANDS: \n
    --train: Trains the algorithm on current environment \n
    --deploy: Runs the policy (actor) network on the current environment\n
    --help: Displays help  
    
    
    """
    )

def pre_process_states(new_obs):
    new_obs = Image.fromarray(new_obs)
    new_obs = new_obs.convert('L')

    new_obs = np.array(new_obs)

    new_obs = new_obs.reshape(1, 96, 96)

    return new_obs
