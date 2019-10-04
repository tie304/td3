import os


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