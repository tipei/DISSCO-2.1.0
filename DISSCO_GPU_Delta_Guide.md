# Running **DISSCO-2.1.0** on NCSA Delta GPU Nodes

This guide explains how to clone, build, and run **DISSCO-2.1.0** inside a CUDA-enabled Apptainer container on the **NCSA Delta GPU cluster**.

---

## 1. Clone the DISSCO Project and Pull the Pre-built CUDA Container (Only needs to be done once)
```bash
git clone https://github.com/tipei/DISSCO-2.1.0.git
apptainer pull dissco_cuda.sif docker://rubin5/dissco-debian12-cuda:12.4
```

---

## 2. Launch an Interactive GPU Session
Request an interactive A100 GPU node and start the container:
```bash
srun --mem=32g      --nodes=1      --ntasks-per-node=1      --cpus-per-task=16      --partition=gpuA100x4-interactive      --account=bbvc-delta-gpu      --gpus-per-node=1      --gpus-per-task=1      --gpu-bind=verbose,per_task:1      --pty apptainer run --nv          --bind /projects/bbvc          ~/dissco_cuda.sif /bin/bash
```

---

## 3. Configure Build Tools Inside the Container and Build (Only needs to be done once)
Once inside the container shell, set the compiler environment and build DISSCO 2.1.0:
```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd DISSCO-2.1.0
premake4
make
```

---

## 4. Run a DISSCO Project Within the Container
Run the `cmod` executable on your `.dissco` file:
```bash
./cmod <path/to/your_project.dissco>
```

---

## 5. Exit the Container
After finishing:
```bash
exit
```

---

### Notes
- The container includes all required dependencies (CUDA toolkit 12.4, GTK, muParser, Xerces-C, sndfile, and Premake4), but doesn't support starting the GUI lassie. It is recommended to build the .dissco locally and then upload it to Delta to run it to save GPU hours.
