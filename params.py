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


if __name__ == "__main__":

    epochs = "2"
    models = "SCaM,JEC_Unet"
    tvt = ["64", "32", "64"]
    labtvt = list()
    
    for argv in sys.argv:
        if argv[:5] == "tvte=" :
            tvte = argv[5:].split(",")
            tvt = tvte[:3]
            epochs = tvte[3]

    for t in tvt:
        if int(t) > 1000 : labtvt.append(f"{int(t/1000)}k")
        else : labtvt.append(t)

    batchs = ["simu", "training", "testing", "analyse"]
    makings = {
        "simu" : f"x={tvt[0]}-{tvt[1]}-{tvt[2]}",
        "training" : f"models={models} train=train{labtvt[0]} valid=valid{labtvt[1]} epoch={epochs}",
        "testing" : f"models={models} train=train{labtvt[0]} test=test{labtvt[2]},test{labtvt[2]}OT,test{labtvt[2]}NL,output_test",
        "analyse" : f"models={models} train=train{labtvt[0]} test=test{labtvt[2]},test{labtvt[2]}OT,test{labtvt[2]}NL,output_test",
    }


    print(f"TVT : {tvt}")
    print(f"Epoch : {epochs}")

    mult = " mult" if "mult" in sys.argv else "" 

    with open(f"making_slurms.zzz", "w") as f:

        for batch in batchs:   

            f.write(f"python TeleBot_Perso/making_bash.py {batch} {makings[batch]}{mult}\n")