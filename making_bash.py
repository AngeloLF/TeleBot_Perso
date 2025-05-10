import os, sys
import coloralf as c
import params


def generate_batch(batch_name, codes, device, mult=True, mail=True, log=True, telegram=[False, "path_token_file", "path_user_id_file"]):

	### ENTETE

	slurm = ["#!/bin/batch"]
	slurm.append(f"#SBATCH --job-name={batch_name}               # Nom du job")



	### Number of task
	ntasks = len(codes) if mult else 1



	### DEVICE
	if device == "cpu":

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_cpu}")

		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=1        # Nombre de CPUs par tâche")
		slurm.append(f"#SBATCH --time=1-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem=4G        # Mémoire demandée")	

	elif device == "gpu":

		slurm.append(f"\n# Description Partition")
		slurm.append(f"#SBATCH --partition={params.partition_gpu}")
		slurm.append(f"#SBATCH --gres=gpu:v100:1")
		
		slurm.append(f"\n# Description de la taches")
		slurm.append(f"#SBATCH --cpus-per-task=5")
		slurm.append(f"#SBATCH --time=1-00:00:00        # Limite de temps")
		slurm.append(f"#SBATCH --ntasks={ntasks}        # Nombre de tâches")
		slurm.append(f"#SBATCH --mem=16G        # Mémoire demandée")

	else:

		print(f"{c.r}WARNING : le device {device} n'existe pas ! Cela doit être : cpu, gpu")



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
	else:

		for i, code in enumerate(codes):
			with open(f"{params.path}/sjob_{i}.sh", "w") as f:
				f.write(f"{params.python} {code}")

		slurm.append(" &\n".join([f"{params.srun} --exclusive sjob_{i}.sh" for i in range(len(codes))]))		
		slurm.append(f"wait")


	### TELEGRAM MESSAGE
	if telegram[0]:
		slurm.append(f"\n# Send a telegram msg")
		msg = f"'BATCH {batch_name} finish'"
		slurm.append(f"{params.python} TeleBot_Perso/telebot.py msg={msg} token={telegram[1]} id={telegram[2]}")



	### WRITING SLURM FILE
	with open(f"{params.path}/{batch_name}.slurm", "w") as f:
		f.write("\n".join(slurm))






def read_SYSargv(argv):

	ARGS = dict()

	for arg in argv:

		if "=" in arg:

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
	args = read_SYSargv(sys.argv[2:])

	if "telegram" not in args.keys() : args["telegram"] = True
	else : args["telegram"] = False if args["telegram"] == "False" else True

	telegram = [args["telegram"], params.telegram_token, params.telegram_user]

	mult = True if "mult" in args.keys() else False



	if batch == "simu_train":
		"""
		x=<nb simu train>-<nb simu valid>
		"""

		batch_name = "simulator_train"

		xtrain, xvalid = args["x"].split("-")
		xtrain, xvalid = int(xtrain), int(xvalid)

		xtrain_label = f"train{int(xtrain/1000)}k" if xtrain > 1000 else f"train{xtrain}"
		xvalid_label = f"valid{int(xvalid/1000)}k" if xvalid > 1000 else f"valid{xvalid}"

		device = "cpu"
		codes = [f"SpecSimulator/alfsimu.py x{xtrain} tsim f={xtrain_label}", f"SpecSimulator/alfsimu.py x{xvalid} tsim f={xvalid_label}"]



	elif batch == "simu_test":
		"""
		x=<nb simu test>

		opt :
		full=<nb_for_lsp> [for lsp test]
		"""

		batch_name = "simulator_test"

		x = int(args["x"])

		x_label = f"test{int(x/1000)}k" if x > 1000 else f"test{x}"

		device = "cpu"
		codes = [f"SpecSimulator/alfsimu.py x{x} tsim f={x_label}", f"SpecSimulator/alfsimu.py x{x} tsim f={x_label}nl noisyless"]

		if 'full' in args.keys():

			codes.append(f"SpecSimulator/alfsimu.py x{args['full']} tsim v=0 lsp test")



	elif batch == "training":
		"""
		models=<model_name1>,<model_name2> ...
		train=<folder_train>
		valid=<folder_valid>
		"""

		device = "gpu"
		models_name = args["models"].split(",")
		batch_name = "training_model"


		# Train folder verification
		if args["train"] not in os.listdir(f"./results/output_simu"):

			error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{args['train']}` unknow"
			print(f"{c.r}{error}{c.d}")
			raise Exception(error)

		# Valid folder verification
		if args["valid"] not in os.listdir(f"./results/output_simu"):

			error = f"WARNING [making_batch.py] : for batch={batch}, folder valid `{args['valid']}` unknow"
			print(f"{c.r}{error}{c.d}")
			raise Exception(error)


		# Add codes
		codes = list()

		for model_name in models_name:

			if model_name in os.listdir(f"./Spec2vecModels"):

				codes.append(f"Spec2vecModels/{model_name}/train_model.py train={args['train']} valid={args['valid']}")

			else:

				error = f"WARNING [making_batch.py] : for batch={batch}, model name `{model_name}` unknow"
				print(f"{c.r}{error}{c.d}")
				raise Exception(error)



	elif batch == "testing":
		"""
		models=<model_name1>,<model_name2> ...
		train=<folder_train>
		test=<folder_test1>,<folder_test2>
		"""

		device = "gpu"
		models_name = args["models"].split(",")
		tests = args["test"].split(",")
		batch_name = "testing_model"


		# Train folder verification
		if args["train"] not in os.listdir(f"./results/output_simu"):

			error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{args['train']}` unknow"
			print(f"{c.r}{error}{c.d}")
			raise Exception(error)

		# Train folder verification
		for test in tests:

			if test not in [*os.listdir(f"./results/output_simu"), *os.listdir(f"./results")]:

				error = f"WARNING [making_batch.py] : for batch={batch}, folder test `{test}` unknow"
				print(f"{c.r}{error}{c.d}")
				raise Exception(error)

		# Add codes
		codes = list()

		for model_name in models_name:

			if model_name in os.listdir(f"./Spec2vecModels"):

				for test in tests:

					codes.append(f"Spec2vecAnalyse/apply_model.py gpu model={model_name} train={args['train']} test={test}")

			else:

				error = f"WARNING [making_batch.py] : for batch={batch}, model name `{model_name}` unknow"
				print(f"{c.r}{error}{c.d}")
				raise Exception(error)


	elif batch == "analyse":
		"""
		models=<model_name1>,<model_name2> ...
		train=<folder_train>
		test=<folder_test1>,<folder_test2>
		"""

		device = "gpu"
		models_name = args["models"].split(",")
		tests = args["test"].split(",")
		batch_name = "analyse_model"


		# Train folder verification
		if args["train"] not in os.listdir(f"./results/output_simu"):

			error = f"WARNING [making_batch.py] : for batch={batch}, folder train `{args['train']}` unknow"
			print(f"{c.r}{error}{c.d}")
			raise Exception(error)

		# Train folder verification
		for test in tests:

			if test not in [*os.listdir(f"./results/output_simu"), *os.listdir(f"./results")]:

				error = f"WARNING [making_batch.py] : for batch={batch}, folder test `{test}` unknow"
				print(f"{c.r}{error}{c.d}")
				raise Exception(error)

		# Add codes
		codes = list()

		for model_name in models_name:

			if model_name in os.listdir(f"./Spec2vecModels"):

				for test in tests:

					codes.append(f"Spec2vecAnalyse/make_graph_result.py model={model_name} train={args['train']} test={test}")

			else:

				error = f"WARNING [making_batch.py] : for batch={batch}, model name `{model_name}` unknow"
				print(f"{c.r}{error}{c.d}")
				raise Exception(error)














	# Construction of SLURM file:

	generate_batch(batch_name, codes, device, telegram=telegram, mult=mult)





