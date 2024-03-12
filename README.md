# Forward-Backward Extended DMD with an Asymptotic Stability Constraint

Companion code for Forward-Backward Extended DMD with an Asymptotic Stability
Constraint

## Installation

To clone the repository and its
[submodule](https://github.com/ramvasudevan/soft-robot-koopman), which contains
the soft robot dataset, run
```sh
$ git clone --recurse-submodules git@github.com:decargroup/forward_backward_koopman.git
```

To install all the required dependencies for this project, it is recommended to create a virtual environment. After activating the virtual environment, run
```sh
(venv) $ pip install -r ./requirements.txt
```

The LMI solver used, MOSEK, requires a license to use. You can request personal
academic license [here](https://www.mosek.com/products/academic-licenses/).

[^1]: On Windows, use `> \venv\Scripts\activate`.
[^2]: On Windows, place the license in `C:\Users\<USER>\mosek\mosek.lic`.

## Usage

To automatically generate all the plots used in the paper, run
```sh
(venv) $ doit
```

The plots will be found in `build/figures/paper`.