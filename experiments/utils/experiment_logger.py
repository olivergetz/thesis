import numpy as np
import datetime
import json
import os

class ExperimentLogger():
    def __init__(self, log_directory : str, experiment_name:str):
        """
        Logs info about an experiment session to a json file. 

        Args:
        - log_directory: where to store the logs.
        - settings: data from a json file that hold hyper-parameter values and layer info.
        """
        print(experiment_name)
        self.log_directory = log_directory
        self.log_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.experiment_name = experiment_name

    def log_results(self, data:dict):
        """
        Args:
            - data: Dictionary with results.
        """
        self.log_time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        self.log = {
            "experiment_name" : self.experiment_name,
            "time": self.log_time,
            "results": data
        }

    def save_log(self):
        
        with open(os.path.join(self.log_directory, f"{self.experiment_name}_{self.log_time}.json"), "w") as f:
            f.write(json.dumps(self.log, cls=NpEncoder))


class NpEncoder(json.JSONEncoder):
    """Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
if __name__ == "__main__":

    with open ('training/model_definitions/vae.json', 'r') as f:
        settings = json.load (f)

    logger = ExperimentLogger("training/generative_experiments/audio_results", settings)