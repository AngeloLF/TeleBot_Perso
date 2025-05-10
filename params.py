import os
path_home = os.path.expanduser("~/")

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"hparameters dont have {name=}.")

partition_cpu = "lsst"
partition_gpu = "gpu_interactive"
python = "python"
srun = "srun"
venv = "./../alfenv"
path = "."
telegram_token = f"{path_home}ccalf.txt"
telegram_user = f"{path_home}alfid.txt"
mail = "angelo.lamure-fontanini@ijclab.in2p3.fr"