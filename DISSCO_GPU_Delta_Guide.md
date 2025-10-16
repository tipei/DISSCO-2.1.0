# Running **DISSCO-2.1.0** on NCSA Delta GPU Nodes

This guide explains how to clone, build, and run **DISSCO-2.1.0** inside a CUDA-enabled Apptainer container on the **NCSA Delta GPU cluster**.

---

## 1. Clone the DISSCO Project
```bash
git clone https://github.com/tipei/DISSCO-2.1.0.git
```

---

## 2. Pull the Pre-built CUDA Container
Pull the pre-built Debian 12 + CUDA 12.4 image from Docker Hub and convert it into an Apptainer `.sif` file:
```bash
apptainer pull dissco_cuda.sif docker://rubin5/dissco-debian12-cuda:12.4
```

---

## 3. Launch an Interactive GPU Session
Request an interactive A100 GPU node and start the container:
```bash
srun --mem=32g      --nodes=1      --ntasks-per-node=1      --cpus-per-task=16      --partition=gpuA100x4-interactive      --account=bbvc-delta-gpu      --gpus-per-node=1      --gpus-per-task=1      --gpu-bind=verbose,per_task:1      --pty apptainer run --nv          --bind /projects/bbvc          ~/dissco_cuda.sif /bin/bash
```

---

## 4. Configure Build Tools Inside the Container
Once inside the container shell, set the compiler environment:
```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

---

## 5. Build DISSCO
Navigate to the cloned project and compile:
```bash
cd DISSCO-2.1.0
premake4
make
```

---

## 6. Run a DISSCO Project
Run the `cmod` executable on your `.dissco` file:
```bash
./cmod <path/to/your_project.dissco>
```

---

## 7. Exit the Container
After finishing:
```bash
exit
```

---

### Notes
- The container includes all required dependencies (CUDA toolkit 12.4, GTK, muParser, Xerces-C, sndfile, and Premake4), but doesn't support starting the GUI lassie. It is recommended to build the .dissco locally and then upload it to Delta to run it to save GPU hours.
