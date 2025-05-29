import os, sys





def extraction_code(code):

	args = code.split(" ")

	params = dict()

	for arg in args:

		if "=" in arg:
			k, v = arg.split("=")
			params[k] = v
		else:
			params[arg] = None

	return params





def extraction(sh, what):

	with open(sh, "r") as f:

		lines = f.read().split("\n")

		for line in lines:

			if "python" in line:

				params = extraction_code(line)

				if what == "training" : inspect_training(params)




def inspect_training(params):

	LR = f"{float(params['lr']):.0e}"

	model_name = f"{params['model']}_{params['loss']}"
	train_name = f"{params['train']}_{LR}"
	load_name = "" if 'load' in params.keys() else f"_{params['load']}"
	epoch = int(params['epoch'])

	nb_make = len(os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/{train_name}{load_name}"))

	print(f"Training {model_name} with {train_name}{load_name} : {nb_make}/{epoch} [{nb_make/epoch*100:.2f} %]")







if __name__ == "__main__":


	what = sys.argv[1]

	shs = [file for file in os.listdir() if file[-3:] == ".sh"]

	for sh in shs : extraction(sh, what)