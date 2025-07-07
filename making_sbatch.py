from types import SimpleNamespace
import os, sys
import params
import coloralf as c


def generate_batch(batch_name, codes, device, mult=False, mail=True, log=True, discobot=False, ext="slurm", mem=None, local=False, gpu_device="v100", nbj="1"):

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
		slurm.append(f"#SBATCH --time={nbj}-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem={memcpu}        # Mémoire demandée")	

	elif device == "gpu" and not local:

		memgpu = "16G" if mem is None else f"{mem}G"

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_gpu}")
		slurm.append(f"#SBATCH --gres=gpu:{gpu_device}:1")
		slurm.append(f"#SBATCH --account={params.account}")
		
		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=5")
		slurm.append(f"#SBATCH --time={nbj}-00:00:00        # Limite de temps")
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

		if "=" in arg:

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



def addJob2args(ARGS, model, loss, train, lr, load, test, score=None):

	ARGS.list_model.append(model)
	ARGS.list_loss.append(loss)
	ARGS.list_train.append(train)
	ARGS.list_lr.append(lr)
	ARGS.list_load.append(load)
	ARGS.list_test.append(test)
	ARGS.nb += 1
	if score is not None : ARGS.list_score.append(score)
	
	return ARGS


def findJob(args, states_path="./results/Spec2vecModels_Results"):

	if "score" not in dir(args):
		print(f"{c.ly}INFO : in findJob, score is not indicated -> def L1,chi2")
		args.score = ["L1", "chi2"]

	if "test" not in dir(args):
		print(f"{c.ly}INFO : in findJob, test is not indicated -> def test1k,test1kOT,test1kExt")
		args.test = ["test1k", "test1kOT", "test1kExt"]

	ARGS_APPLY = SimpleNamespace()
	ARGS_ANALYSE = SimpleNamespace()


	for ARGS in [ARGS_APPLY, ARGS_ANALYSE]:

		ARGS.list_model = list()
		ARGS.list_loss = list()
		ARGS.list_train = list()
		ARGS.list_lr = list()
		ARGS.list_load = list()
		ARGS.list_test = list()
		ARGS.list_score = list()
		ARGS.nb = 0


	for modelwl in args.modelwl:
		
		model, loss = modelwl.split("_")
		print(f"\n{c.y}Analyseof model {model} [{loss}]{c.d}")

		for state in os.listdir(f"{states_path}/{modelwl}/states"):

			if "_best.pth" in state:

				subs = state[:-9]
					
				if subs.count("_") == 1: 
					train, lr = subs.split("_")
					load = "None"
					pred_folder = f"pred_{model}_{loss}_{train}_{lr}"
				elif subs.count("_") == 3:
					loadtrain, loadlr, train, lr = subs.split("_")
					load = f"{loadtrain}_{loadlr}"
					pred_folder = f"pred_{model}_{loss}_{load}_{train}_{lr}"
				else:
					raise Exception(f"WARNING [making_sbatch.py] : State {state} counts `_` not in [1, 3]")

				print(f"{c.ly}Analyseof : detect {subs} -> {train} > {lr} > {load}{c.d}")

				for otest in args.test:

					test = otest if "no0" not in train else f"{otest}no0"

					if pred_folder in os.listdir(f"./results/output_simu/{test}"):
						print(f"{c.lg}- Test {test} apply{c.d}")

						for score in args.score:

							if pred_folder in os.listdir(f"./results/analyse/{score}") and test in os.listdir(f"./results/analyse/{score}/{pred_folder}") and "resume.txt" in os.listdir(f"./results/analyse/{score}/{pred_folder}/{test}"):
								print(f"{c.lg}  |-- score {score} analyse{c.d}")
							else:
								print(f"{c.lr}  |-- score {score} not analyse{c.d}")
								ARGS_ANALYSE = addJob2args(ARGS_ANALYSE, model, loss, train, lr, load, test, score)


					else:
						print(f"{c.r}- Test {test} not apply{c.d}")
						ARGS_APPLY = addJob2args(ARGS_APPLY, model, loss, train, lr, load, test)



	choice = None

	while choice not in ["", "analyse", "apply"]:

		print(f"Number of job `apply`   : {ARGS_APPLY.nb}")
		print(f"Number of job `analyse` : {ARGS_ANALYSE.nb}")
		choice = input(f"Make anything ? (an/analyse or ap/apply) : ")

		if choice in ["an", "analyse"] : choice = "analyse"
		elif choice in ["ap", "apply"] : choice = "apply" 

	args.ARGS_APPLY = ARGS_APPLY
	args.ARGS_ANALYSE = ARGS_ANALYSE
	args.findjob_choice = choice

	return args








if __name__ in "__main__":


	batch_codes = {
		"flash"     : ["None",                            ["jobname", "code"]],
		"simu"      : ["SpecSimulator/alfsimu.py",        ["x", "name", "set", "simup"]],
		"training"  : ["Spec2vecModels/train_models.py",  ["model", "loss", "train", "lr", "valid", "e"]],
		"apply"     : ["Spec2vecAnalyse/apply_model.py",  ["model", "loss", "train", "lr", "test"]],
		"analyse"   : ["Spec2vecAnalyse/analyse_test.py", ["model", "loss", "train", "lr", "test", "score"]],
		"findjob"   : ["None",                            ["modelwl"]] # Model with loss like `SCaM_chi2`
	}

	arg2split = ["model", "modelwl", "loss", "train", "test", "lr", "load", "x", "name", "set", "score", "simup"]

	batch, args = read_SYSargv(batch_codes, arg2split)
	if batch == "findjob" : args = findJob(args)

	args.discobot = True if "discobot" in dir(args) else False
	args.mult = False if "nomult" in dir(args) else True
	args.local = True if "local" in dir(args) else False
	if "load" not in dir(args) : args.load = ["None"]
	if "mem" not in dir(args) : args.mem = None
	if "gd" not in dir(args) : args.gd = "v100"
	if "nbj" not in dir(args) : args.nbj = "1"

	batch_names = list()
	codes = list()
	make_jobs = True



	if batch == "flash":

		device = "cpu"

		codes.append(args.code)
		batch_names.append(args.jobname)



	elif batch == "simu":

		device = "cpu"

		for ni, xi, si, simupi in zip(args.name, args.x, args.set, args.simup):

			if "output_test" != ni:
				
				codes.append(f"{batch_codes['simu'][0]} x{xi} f={ni} {si} {simupi} tsim")
				batch_names.append(f"{batch}_{ni}")

			else:

				codes.append(f"{batch_codes['simu'][0]} x{xi} tsim v=0 lsp test")
				batch_names.append(f"{batch_name}_output_test")



	elif batch == "findjob" and args.findjob_choice == "apply":

		device = "cpu"

		for model, loss, train, lr, load, test in zip(args.ARGS_APPLY.list_model, args.ARGS_APPLY.list_loss, args.ARGS_APPLY.list_train, args.ARGS_APPLY.list_lr, args.ARGS_APPLY.list_load, args.ARGS_APPLY.list_test):

			codes.append(f"{batch_codes['apply'][0]} model={model} loss={loss} train={train} test={test} lr={lr} load={load}")
			batch_names.append(f"apply_{model}_{loss}_{train}_{test}_{lr}_{load}")



	elif batch == "findjob" and args.findjob_choice == "analyse":

		device = "cpu"

		for model, loss, train, lr, load, test, score in zip(args.ARGS_ANALYSE.list_model, args.ARGS_ANALYSE.list_loss, args.ARGS_ANALYSE.list_train, args.ARGS_ANALYSE.list_lr, args.ARGS_ANALYSE.list_load, args.ARGS_ANALYSE.list_test, args.ARGS_ANALYSE.list_score):
		
			codes.append(f"{batch_codes['analyse'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
			batch_names.append(f"analyse_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")



	elif batch == "findjob":

		make_jobs = False



	else:

		for model in args.model:

			# Check model
			if f"{model}.py" not in os.listdir(f"./Spec2vecModels/architecture") and model != "Spectractor":
				raise Exception(f"The architecture model {model} unknow")


			for loss in args.loss:


				for train in args.train:

					# Check train
					if train not in os.listdir(f"./results/output_simu") and model != "Spectractor":
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

										device = "cpu" if "gpu" not in sys.argv else "gpu"

										codes.append(f"{batch_codes['apply'][0]} model={model} loss={loss} train={train} test={test} lr={lr} load={load} {device}")
										batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{load}")

									elif batch == "analyse":

										device = "cpu"

										for score in args.score:

											codes.append(f"{batch_codes['analyse'][0]} model={model} train={train} test={test} loss={loss} lr={lr} score={score} load={load}")
											batch_names.append(f"{batch}_{model}_{loss}_{train}_{test}_{lr}_{score}_{load}")



	# Construction of SLURM file:

	if make_jobs:

		if not args.mult:

			extsup = "slurm" if not args.local else "ps1"
			generate_batch(batch, codes, device, discobot=args.discobot, mem=args.mem, ext=extsup, local=args.local, gpu_device=args.gd, nbj=args.nbj)

		else:

			extsup = "slurm" if not args.local else "ps1"
			ext = "sh" if not args.local else "ps1"

			with open(f"{batch}.{extsup}", "w") as f:
				for name, code in zip(batch_names, codes):
					generate_batch(name, code, device, discobot=args.discobot, ext=ext, mem=args.mem, local=args.local, gpu_device=args.gd, nbj=args.nbj)
					f.write(f"sbatch {params.path}/{name}.sh\n")
									




