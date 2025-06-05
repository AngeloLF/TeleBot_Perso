from types import SimpleNamespace
import os, sys
import params



def generate_batch(batch_name, codes, device, mult=False, mail=True, log=True, discobot=False, ext="slurm", mem=None, local=False):

	### ENTETE
	if not local:
		slurm = ["#!/bin/bash"]
		slurm.append(f"#SBATCH --job-name={batch_name}               # Nom du job")
	else:
		slurm = ["Write-Host \"PS1 : sbatch\""]


	### Number of task
	ntasks = len(codes) if mult else 1



	### DEVICE
	if device == "cpu" and not local:

		memcpu = "4G" if mem is None else f"{mem}G" 

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_cpu}")
		slurm.append(f"#SBATCH --account={params.account}")

		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=1        # Nombre de CPUs par tâche")
		slurm.append(f"#SBATCH --time=1-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem={memcpu}        # Mémoire demandée")	

	elif device == "gpu" and not local:

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

	elif not local:

		print(f"WARNING : le device {device} n'existe pas ! Cela doit être : cpu, gpu")



	### MAIL

	if mail and not local:
		slurm.append(f"\n#SBATCH --mail-user={params.mail}")
		slurm.append(f"#SBATCH --mail-type=BEGIN,END,FAIL")



	### OUTPUT
	if log and not local:
		slurm.append(f"\n#SBATCH --output=slurm_log/slurm_out/{batch_name}_%j.txt        # Fichier de sortie")
		slurm.append(f"#SBATCH --error=slurm_log/slurm_err/{batch_name}_%j.txt        # Fichier d'erreur")



	### V-ENV
	if params.venv is not None and not local:
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





def read_SYSargv(batch_codes, arg2split):


	ARGS = SimpleNamespace()
	batch = sys.argv[1]
	argv = sys.argv[2:]


	if batch not in batch_codes.keys(): 
		if batch == "help": 
			print(arg2split)
			sys.exit()
		else: 
			raise Exception(f"BATCH {batch} unknow")


	for arg in argv:

		if "=" in arg and ".py" not in arg:

			k, v = arg.split("=")

			if k in arg2split : ARGS.__setattr__(k, v.split(","))
			else : ARGS.__setattr__(k, v) 

		else:

			ARGS.__setattr__(arg, None)


	errors = list()
	for attr in batch_codes[batch][1]:
		if attr not in dir(ARGS) : errors.append(attr)

	if len(errors) > 0 : raise Exception(f"{', '.join(errors)} missing")


	return batch, ARGS




if __name__ in "__main__":


	batch_codes = {
		"simu"     : ["SpecSimulator/alfsimu.py",        ["x", "name", "set", "simup"]],
		"training" : ["Spec2vecModels/train_models.py",  ["model", "loss", "train", "lr", "valid", "e"]],
		"apply"    : ["Spec2vecAnalyse/apply_model.py",  ["model", "loss", "train", "lr", "test"]],
		"analyse"  : ["Spec2vecAnalyse/analyse_test.py", ["model", "loss", "train", "lr", "test", "score"]],
	}

	arg2split = ["model", "loss", "train", "test", "lr", "load", "x", "name", "set", "score", "simup"]

	batch, args = read_SYSargv(batch_codes, arg2split)

	args.discobot = True if "discobot" in dir(args) else False
	args.mult = False if "nomult" in dir(args) else True
	args.local = True if "local" in dir(args) else False
	if "load" not in dir(args) : args.load = ["None"]
	if "mem" not in dir(args) : args.mem = None

	batch_names = list()
	codes = list()


	if batch == "simu":

		device = "cpu"

		for ni, xi, si, simupi in zip(args.name, args.x, args.set, args.simup):

			if "output_test" != ni:
				
				codes.append(f"{batch_codes['simu'][0]} x{xi} f={ni} {si} {simupi} tsim")
				batch_names.append(f"{batch}_{ni}")

			else:

				codes.append(f"{batch_codes['simu'][0]} x{xi} tsim v=0 lsp test")
				batch_names.append(f"{batch_name}_output_test")


	else:

		for model in args.model:

			# Check model
			if f"{model}.py" not in os.listdir(f"./Spec2vecModels/architecture"):
				raise Exception(f"The architecture model {model} unknow")


			for loss in args.loss:


				for train in args.train:

					# Check train
					if train not in os.listdir(f"./results/output_simu"):
						raise Exception(f"Train folder {train} not in ./results/output_simu")


					for lr in args.lr:


						for load in args.load:


							if batch == "training" and args.valid in os.listdir(f"./results/output_simu"):

								device = "gpu"

								codes.append(f"{batch_codes['training'][0]} model={model} loss={loss} train={train} valid={args.valid} epoch={args.e} lr={lr} load={load}")
								batch_names.append(f"{batch}_{model}_{loss}_{train}_{lr}_{load}")

							elif batch == "training" and args.valid not in os.listdir(f"./results/output_simu"):

								# Check valid
								raise Exception(f"Valid folder {args.valid} not in ./results/output_simu")

							else:

								for test in args.test:

									# Check test
									if test not in os.listdir(f"./results/output_simu"):
										raise Exception(f"Test folder {test} not in ./results/output_simu")


									if batch == "apply":

										device = "cpu"

										codes.append(f"{batch_codes['apply'][0]} model={model} loss={loss} train={train} test={test} lr={lr} load={load}")
										batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{load}")

									elif batch == "analyse":

										device = "cpu"

										for score in args.score:

											codes.append(f"{batch_codes['analyse'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
											batch_names.append(f"{batch}_{model}_{train}_{test}_{lr}_{score}_{load}")



	# Construction of SLURM file:

	if not args.mult:

		extsup = "slurm" if not args.local else "ps1"
		generate_batch(batch, codes, device, discobot=args.discobot, mem=args.mem, ext=extsup, local=args.local)

	else:

		extsup = "slurm" if not args.local else "ps1"
		ext = "sh" if not args.local else "ps1"

		with open(f"{batch}.{extsup}", "w") as f:
			for name, code in zip(batch_names, codes):
				generate_batch(name, code, device, discobot=args.discobot, ext=ext, mem=args.mem, local=args.local)
				f.write(f"sbatch {params.path}/{name}.sh\n")
									




