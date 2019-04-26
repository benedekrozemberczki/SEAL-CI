from utils import tab_printer
from parser import parameter_parser
from seal import SEALTrainer
def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SEALTrainer(args)
    embeddings = trainer.fit()
    trainer.score(embeddings)
    
if __name__ == "__main__":
    main()
