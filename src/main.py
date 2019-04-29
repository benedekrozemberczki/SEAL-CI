import torch
from utils import tab_printer
from seal import SEALCITrainer
from parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SEAL-CI model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SEALCITrainer(args)
    trainer.fit()
    trainer.score()
    
if __name__ == "__main__":
    main()
