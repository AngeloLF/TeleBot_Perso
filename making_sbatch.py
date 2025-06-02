from types import SimpleNamespace
import os, sys
import params



def generate_batch(batch_name, codes, device, mult=False, mail=True, log=True, discobot=False, ext="slurm", mem=None):

	### ENTETE

	slurm = ["#!/bin/bash"]
	slurm.append(f"#SBATCH --job-name={batch_name}               # Nom du job")



	### Number of task
	ntasks = len(codes) if mult else 1



	### DEVICE
	if device == "cpu":

		memcpu = "4G" if mem is None else f"{mem}G" 

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_cpu}")
		slurm.append(f"#SBATCH --account={params.account}")

		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=1        # Nombre de CPUs par tâche")
		slurm.append(f"#SBATCH --time=1-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem={memcpu}        # Mémoire demandée")	

	elif device == "gpu":

		memgpu = "16G" if mem is None else f"{mem}G" 

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_gpu}")
		slurm.append(f"#SBATCH --gres=gpu:v100:1")
		slurm.append(f"#SBATCH --account={params.account}")
		
		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=5")
		slurm.append(f"#SBATCH --time=1-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem={memgpu}        # Mémoire demandée")

	else:

		print(f"WARNING : le device {device} n'existe pas ! Cela doit être : cpu, gpu")



	### MAIL

	if mail:
		slurm.append(f"\n#SBATCH --mail-user={params.mail}")
		slurm.append(f"#SBATCH --mail-type=BEGIN,END,FAIL")



	### OUTPUT
	if log:
		slurm.append(f"\n#SBATCH --output=slurm_log/slurm_out/{batch_name}_%j.txt        # Fichier de sortie")
		slurm.append(f"#SBATCH --error=slurm_log/slurm_err/{batch_name}_%j.txt        # Fichier d'erreur")



	### V-ENV
	if params.venv is not None:
		slurm.append(f'\nVENV_PATH="{params.venv}"')
		slurm.append('source ${' + 'VENV_PATH' + '}/bin/activate')



	### code
	slurm.append(f"\n# Codes to cumpute :")

	if isinstance(codes, (str)):
		codes = [codes]

	if not mult:
		slurm.append("\n".join([f"{params.python} {code}" for code in codes]))
		print(f"{params.python} {codes}")
	else:

		for i, code in enumerate(codes):
			with open(f"{params.path}/sjob_{i}.sh", "w") as f:
				f.write(f"{params.python} {code}")

		slurm.append(" &\n".join([f"{params.srun} --exclusive sjob_{i}.sh" for i in range(len(codes))]))		
		slurm.append(f"wait")



	### TELEGRAM MESSAGE
	if discobot:
		slurm.append(f"\n# Send a discord msg")
		msg = f"'BATCH {batch_name} finish'"
		slurm.append(f"{params.python} TeleBot_Perso/discobot.py msg={msg}")



	### WRITING SLURM FILE
	with open(f"{params.path}/{batch_name}.{ext}", "w") as f:
		f.write("\n".join(slurm))





def read_SYSargv(batch_dispo, arg2split):


	ARGS = SimpleNamespace()
	batch = sys.argv[1]
	argv = sys.argv[2:]


	if batch not in batch_dispo : raise Exception(f"BATCH {batch} unknow")


	for arg in argv:

		if "=" in arg and ".py" not in arg:

			k, v = arg.split("=")

			if k in arg2split : ARGS.__setattribute__(k, v.split(","))
			else : ARGS.__setattribute__(k, v) 

		else:

			ARGS.__setattribute__(arg, None)

	return batch, ARGS




if __name__ in "__main__":


	batch_codes = {
		"simu"     : "SpecSimulator/alfsimu.py",
		"training" : "Spec2vecModels/train_models.py",
		"apply"    : "Spec2vecAnalyse/apply_model.py",
		"analyse"  : "Spec2vecAnalyse/analyse_test.py",
	}

	arg2split = ["model", "loss", "train", "test", "lr", "load", "x", "name", "set"]

	batch, args = read_SYSargv(list(batch_codes.keys()), arg2split)

	if "discobot" in dir(args) : args.discobot = True
	else : args.discobot = False

	args.discobot = True if "discobot" in dir(args) else False
	args.mult = False if "nomult" in dir(args) else True
	if "load" not in dir(args) : args.load = None
	if "mem" not in dir(args) : args.mem = None


	batch_names = list()
	codes = list()


	if batch == "simu":

		for ni, xi, si in zip(args.name, args.x, args.set):

			if "output" not in ni:
				
				codes.append(f"SpecSimulator/alfsimu.py x{xi} f={ni} si tsim")
				batch_names.append(f"{batch}_{ni}")

			else:

				


	else:




