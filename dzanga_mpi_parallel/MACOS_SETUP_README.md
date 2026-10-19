## MPI Setup

The project requires both `mpi4py` and a system MPI implementation.

### macOS (Apple Silicon or Intel)

Install Open MPI using Homebrew:

```bash
brew install open-mpi
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements_mpi.txt
```

Verify the MPI installation:

```bash
python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

### Ubuntu / Debian Linux

Install Open MPI and its development libraries:

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements_mpi.txt
```

Verify the MPI installation:

```bash
python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```
