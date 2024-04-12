import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
num_arrays = 22
arrays = [[] for _ in range(num_arrays)]
file_path = 'log'  # Replace 'your_file.txt' with the actual file path
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line
for i in range(len(lines)):
    line = lines[i].strip()
    for num in range(num_arrays):
        if " "+str(num + 340) in line:
            # Extract the number from the next line and add it to the corresponding array
            value = lines[i+1].strip().split()[1]
            arrays[num].append(float(value))
spec2017=["perlbench_r", "mcf_r", "xalancbmk_r",\
            "deepsjeng_r", "leela_r", "exchange2_r",\
            "xz_r", "bwaves_r", "cactuBSSN_r", "namd_r",\
            "lbm_r", "blender_r", "cam4_r", "nab_r",\
            "fotonik3d_r", "roms_r",
            "xalancbmk_s", "leela_s", "exchange2_s",\
            "povray_r","wrf_r","imagick_r"]
plt.figure(figsize=(12, 9))

plt.boxplot(arrays, labels=spec2017)
#plt.xlabel('Arrays')
plt.ylabel('MAPE')
plt.xticks(rotation=35, ha="right")
plt.title('Boxplot for SPEC CPU 2017')
plt.savefig('boxplot_17')
