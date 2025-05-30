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






def read_SYSargv(argv):

	ARGS = dict()

	for arg in argv:

		if "=" in arg and ".py" not in arg:

			k, v = arg.split("=")
			ARGS[k] = v

		else:

			ARGS[arg] = None

	return ARGS





if __name__ == "__main__":
	"""
	argv : batch

	optionnal:
		venv=<***> : path to virtual environnement
		telegram=False for not send telegram msg
	"""

	batch = sys.argv[1]
	args = read_SYSargv(sys.argv[2:]) if batch != "flash" else read_SYSargv(sys.argv[3:])

	
	discobot = True if "discobot" in args.keys() else False
	mult = True if "mult" in args.keys() else False
	mem = None if "mem" not in args.keys() else args["mem"]

	batch_names = list()



	if batch == "flash" :

		device = "gpu" if "gpu" in args.keys() else "cpu"
		batch_name = "flash" if "name" not in args.keys() else args["name"]

		codes = [sys.argv[2]]



	elif batch == "simu":
		"""
		x=<nb simu train>-<nb simu valid>-<nb simu test>

		opt :
		full=<nb_for_lsp> [for lsp test]
		"""

		batch_name = "simulator"
		device = "cpu"

		xtrain, xvalid, xtest = args["x"].split("-")
		xtrain, xvalid, xtest = int(xtrain), int(xvalid), int(xtest)

		xtrain_label = f"train{int(xtrain/1000)}k" if xtrain > 1000 else f"train{xtrain}"
		xvalid_label = f"valid{int(xvalid/1000)}k" if xvalid > 1000 else f"valid{xvalid}"
		xtest_label  = f"test{int(xtest/1000)}k" if xtest > 1000 else f"test{xtest}"

		codes = [f"SpecSimulator/alfsimu.py x{xtrain} tsim f={xtrain_label}",
			     f"SpecSimulator/alfsimu.py x{xvalid} tsim f={xvalid_label}",
			     f"SpecSimulator/alfsimu.py x{xtest} tsim f={xtest_label}",
			     f"SpecSimulator/alfsimu.py x{xtest} tsim f={xtest_label}Ext test",
			     f"SpecSimulator/alfsimu.py x{xtest} tsim f={xtest_label}OT set1"]
		batch_names = [f"{batch_name}_train", f"{batch_name}_valid", f"{batch_name}_test", f"{batch_name}_testExt", f"{batch_name}_testOT"]


		if 'full' in args.keys():

			codes.append(f"SpecSimulator/alfsimu.py x{args['full']} tsim v=0 lsp test")
			batch_names.append(f"{batch_name}_full")



	elif batch == "training":
		"""
		models=<model_name1>,<model_name2> ...
		train=<folder_train1>,<folder_train2> ...
		valid=<folder_valid>
		epoch=<num_epoch>
		lr=<lr1>,<lr2> ...
		loss=<loss1>,<loss2> ...
		load=<load1>,<load2> ...
		"""

		device = "gpu"
		models_name = args["models"].split(",")
		trains = args["train"].split(",")
		lrs = args["lr"].split(",")
		losses = args["loss"].split(",")
		batch_name = "training_model"

		loads = None if "load" not in args.keys() else args["load"].split(",")


		# Train folder verification
		for train in trains:
			if train not in os.listdir(f"./results/output_simu"):

				error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{train}` unknow"
				print(f"{error}")
				# raise Exception(error)

		# Valid folder verification
		if args["valid"] not in os.listdir(f"./results/output_simu"):

			error = f"WARNING [making_batch.py] : for batch={batch}, folder valid `{args['valid']}` unknow"
			print(f"{error}")
			# raise Exception(error)


		# Add codes
		codes = list()

		for model_name in models_name:

			if f"{model_name}.py" in os.listdir(f"./Spec2vecModels/architecture"):

				for lr in lrs:

					for train in trains:

						for loss in losses:

							if loads is None:

								codes.append(f"Spec2vecModels/train_models.py model={model_name} train={train} valid={args['valid']} epoch={args['epoch']} lr={lr} loss={loss}")
								batch_names.append(f"{batch_name}_{model_name}_{loss}_{train}_{lr}")

							else:

								for load in loads:
									codes.append(f"Spec2vecModels/train_models.py model={model_name} train={train} valid={args['valid']} epoch={args['epoch']} lr={lr} loss={loss} load={load}")
									batch_names.append(f"{batch_name}_{model_name}_{loss}_{train}_{lr}_{load}")

			else:

				error = f"WARNING [making_batch.py] : for batch={batch}, model architecture `{model_name}` unknow"
				print(f"{error}")
				# raise Exception(error)



	elif batch == "apply":
		"""
		models=<model_name1>,<model_name2> ...
		loss=<loss1>,<loss2> ...
		train=<folder_train>,<folder_train> ...
		test=<folder_test1>,<folder_test2> ...
		lr=<lr1>,<lr2> ...
		load=<load1>,<load2> ...
		"""

		device = "gpu" if "gpu" in args.keys() else "cpu"
		models_name = args["models"].split(",")
		losses = args["loss"].split(",")
		trains = args["train"].split(",")
		tests = args["test"].split(",")
		lrs = args["lr"].split(",")
		batch_name = "apply_model"

		loads = None if "load" not in args.keys() else args["load"].split(",")


		# Train folder verification
		for train in trains:
			if train not in os.listdir(f"./results/output_simu"):

				error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{train}` unknow"
				print(f"{error}")


		# Train folder verification
		for test in tests:
			if test not in [*os.listdir(f"./results/output_simu"), *os.listdir(f"./results")]:

				error = f"WARNING [making_batch.py] : for batch={batch}, folder test `{test}` unknow"
				print(f"{error}")


		# Add codes
		codes = list()

		for model_name in models_name:

			for loss in losses:

				for lr in lrs:

					for train in trains:

						for test in tests:

							if loads is None:

								codes.append(f"Spec2vecAnalyse/apply_model.py model={model} loss={loss} train={train} test={test} lr={lr}")
								batch_names.append(f"{batch_name}_{model}_{loss}_{train}_{test}_{lr}")

							else:
								for load in loads:

									codes.append(f"Spec2vecAnalyse/apply_model.py model={model} loss={loss} train={train} test={test} lr={lr} load={load}")
									batch_names.append(f"{batch_name}_{model}_{loss}_{train}_{test}_{lr}_{load}")


	elif batch == "analyse":
		"""
		models=<model_name1>,<model_name2> ...
		train=<folder_train>,<folder_train> ...
		test=<folder_test1>,<folder_test2> ...
		lr=<lr1>,<lr2> ...
		score=<score1>,<score2> ...
		"""

		device = "cpu"
		models_name = args["models"].split(",")
		trains = args["train"].split(",")
		tests = args["test"].split(",")
		lrs = args["lr"].split(",")
		scores = args["score"].split(",")
		batch_name = "analyse_model"


		# Train folder verification
		for train in trains:
			if train not in os.listdir(f"./results/output_simu"):

				error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{train}` unknow"
				print(f"{error}")


		# Train folder verification
		for test in tests:
			if test not in [*os.listdir(f"./results/output_simu"), *os.listdir(f"./results")]:

				error = f"WARNING [making_batch.py] : for batch={batch}, folder test `{test}` unknow"
				print(f"{error}")


		# Add codes
		codes = list()

		for model_name in models_name:

			if model_name in os.listdir(f"./results/Spec2vecModels_Results"):

				for lr in lrs:

					for train in trains:

						for test in tests:

							for score in scores:

								codes.append(f"Spec2vecAnalyse/analyse_test.py model={model_name} train={train} test={test} lr={lr} score={score}")
								batch_names.append(f"{batch_name}_{model_name}_{train}_{test}_{lr}_{score}")

			else:

				error = f"WARNING [making_batch.py] : for batch={batch}, model name `{model_name}` unknow"
				print(f"{error}")
				# raise Exception(error)



	else :

		if batch in "help":

			pass


		raise Exception(f"Batch {batch} unknow")






	# Construction of SLURM file:

	if not mult:

		generate_batch(batch_name, codes, device, discobot=discobot, mem=mem)

	else:

		with open(f"{batch_name}.slurm", "w") as f:
			for name, code in zip(batch_names, codes):
				generate_batch(name, code, device, discobot=discobot, ext="sh", mem=mem)
				f.write(f"sbatch {params.path}/{name}.sh\n")





