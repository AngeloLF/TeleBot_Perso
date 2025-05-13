import os, sys
path_home = os.path.expanduser("~/")

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"hparameters dont have {name=}.")

partition_cpu = "lsst,htc"
account = "lsst"
partition_gpu = "gpu_interactive"
python = "python"
srun = "srun"
venv = "./../alfenv"
path = "."
telegram_token = f"{path_home}ccalf.txt"
telegram_user = f"{path_home}alfid.txt"
mail = "angelo.lamure-fontanini@ijclab.in2p3.fr"


models = "SCaM,JEC_Unet"
tvt = ["64", "32", "64"]

batchs = ["simu", "training", "testing", "analyse"]
makings = {
    "simu" : f"x={tvt[0]}-{tvt[1]}-{tvt[2]}",
    "training" : f"models={models} train=train{tvt[0]} valid=valid{tvt[1]}",
    "testing" : f"models={models} train=train{tvt[0]} test=test{tvt[2]},test{tvt[2]}OT,test{tvt[2]}NL,output_test",
    "analyse" : f"models={models} train=train{tvt[0]} test=test{tvt[2]},test{tvt[2]}OT,test{tvt[2]}NL,output_test",
}


if __name__ == "__main__":

    mult = " mult" if "mult" in sys.argv else "" 

    with open(f"making_batch", "w") as f:

        for batch in batchs:   

            f.write(f"python TeleBot_Perso/making_bash.py {batch} {makings[batch]}{mult}\n")