import sys 
import numpy as np

PRETRAIN_ITERS = 50000
LATENT_CODE_ITERS = 1250
DIP_ITERS = 1000

def get_pretrain(fpath):
	lines = []

	with open(fpath, "r") as f:
		start = False
		end = False
		for line in f.readlines():
			if line.strip() == "Pretraining":
				start = True
				continue

			if line.strip() == "":
				end = True

			if start and end:
				break

			if start and not end:
				line = line.strip()
				if line[:8] == "time per":
					line = float(line[line.find(":")+1:-1])
					lines.append(line)

	return lines[1:] # skip first log

def get_pretrain(fpath):
	kw = "_pretrain total time"
	with open(fpath, "r") as f:
		for line in f.readlines():
			line = line.strip()
			if line[:len(kw)] == kw:
				return float(line[len(kw)+1:-1])
	return 0

def get_latent_code(fpath):
	lines = []
	kw = "estimate_initial_latent_codes total time"
	with open(fpath, "r") as f:
		for line in f.readlines():
			line = line.strip()
			if line[:len(kw)] == kw:
				line = float(line.split(" ")[-1][:-1])
				lines.append(line)
	return lines

def get_dip(fpath):
	lines = []

	if "15shots" in fpath:
		total_times = []
		kw = "Time"
		with open(fpath, "r") as f:
			for line in f.readlines():
				line = line.strip()
				if line[:len(kw)] == kw:
					line = float(line.split(" ")[1])
					total_times.append(line)

		pretrain_times = get_pretrain(fpath)
		init_times = get_latent_code(fpath)

		kw1 = "in"
		kw2 = "seconds"
		load_times = []
		with open(fpath, "r") as f:
			for line in f.readlines():
				line = line.strip()
				if line[:len(kw1)] == kw1 and line[-len(kw2):] == kw2:
					line = float(line.split(" ")[1])
					load_times.append(line)

		if pretrain_times:
			init_times[0] += pretrain_times

		if len(total_times) != len(init_times) and len(init_times) != len(load_times):
			print("Tot", len(total_times),total_times)
			print("init", len(init_times),init_times)
			print("load", len(load_times),load_times)
		else:
			for i in range(len(load_times)):
				init_times[i] += load_times[i]


		lines = [tot - init for tot, init in zip(total_times, init_times)]

	elif "0shots" in fpath:
		kw = "Time"
		with open(fpath, "r") as f:
			for line in f.readlines():
				line = line.strip()
				if line[:len(kw)] == kw:
					line = float(line.split(" ")[1])
					lines.append(line)
	return lines

def main():

	if len(sys.argv) < 2:
		raise ValueError("Expect path to .log file as cmdline arg")

	fpath = sys.argv[1]
	pretrain_times = get_pretrain(fpath)
	latent_code_times = get_latent_code(fpath)
	dip_times = get_dip(fpath)

	if pretrain_times:
		print(f"Pretrain {pretrain_times / PRETRAIN_ITERS} (sec/iteration)")
		print(f"Pretrain {pretrain_times} (sec/iteration)")
	if latent_code_times:
		print(f"Init latent code {np.mean(latent_code_times)} +/- {np.std(latent_code_times)} (seconds total) (N={len(latent_code_times)})")
		print(f"Init latent code {np.mean(latent_code_times)/LATENT_CODE_ITERS} +/- {np.std(latent_code_times)/LATENT_CODE_ITERS} (sec/iteration) (N={len(latent_code_times)})")
	if dip_times:
		print(f"DIP {np.mean(dip_times)} +/- {np.std(dip_times)} (seconds total) (N={len(dip_times)})")
		print(f"DIP {np.mean(dip_times)/DIP_ITERS} +/- {np.std(dip_times)/DIP_ITERS} (sec/iteration) (N={len(dip_times)})")

if __name__ == "__main__":
	main()