import json
import sys
from modules.train import Train
from modules.deploy import Deploy
from utils.utils import display_help

with open('parameters.json') as f:
    parameters = json.load(f)

try:
    if sys.argv[1] == "--train":
        session = Train(parameters)
        session.train()
    elif sys.argv[1] == "--deploy":
        session = Deploy(parameters)
        session.run_policy()
    elif sys.argv[1] == "--help":
        print(display_help())
    else:
        print(display_help())
except IndexError:
    print(display_help())
finally:
    pass






