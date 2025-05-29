import os, sys
import coloralf as c




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

	if "load" in params.keys():

		pre_train, pre_lr = params['load'].split('_')
		pre_LR = f"{float(pre_lr):.0e}"
		load_name = f"{pre_train}_{pre_LR}_"

	else:

		load_name = ""

	model_name = f"{params['model']}_{params['loss']}"
	train_name = f"{params['train']}_{LR}"
	epoch = int(params['epoch'])

	nb_make = len(os.listdir(f"./results/Spec2vecModels_Results/{model_name}/epoch/{load_name}{train_name}"))
	lmax = len(str(epoch))

	if   nb_make == epoch : color = c.lg
	elif nb_make > epoch * 0.8 : color = c.ly
	elif nb_make > 0 : color = c.lr
	else : color = c.lk

	print(f"Training {model_name} with {load_name}{train_name} : {color}{nb_make:{lmax}}/{epoch}{c.d} [{nb_make/epoch*100:6.2f} %]")







if __name__ == "__main__":


	what = sys.argv[1]

	shs = [file for file in os.listdir() if file[-3:] == ".sh"]

	for sh in shs : extraction(sh, what)