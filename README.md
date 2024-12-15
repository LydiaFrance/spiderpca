# spiderpca

Running PCA on spider biomechanics data

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- git (for installation from source)

### Installation

The code has been packaged as a Python package, so you can install it using pip.

First, clone the repository in a suitable directory:

```bash
git clone https://github.com/LydiaFrance/spiderpca
cd spiderpca
```

Next, create a new environment. I recommend venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

If you prefer conda, you can do this instead:

```bash
conda create -n spiderpca python=3.10
conda activate spiderpca
```

Then install the package (and update pip):

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

Navigate to the `examples/` directory and open the desired notebook with Jupyter or JupyterLab. 

## Data

I have not included the data in the repository, but you should put it in the `data/` directory.

## Walkthrough Notebooks

The functions used in the notebooks are in the `src/spiderpca` directory.

### 1. Data Preparation and Visualization
[`01_prep_and_visualise.ipynb`](examples/01_prep_and_visualise.ipynb)
- Learn how to load and preprocess spider movement data
- Visualise raw movement patterns
- Prepare data for PCA analysis

### 2. Full Shape PCA Analysis
[`02_full_shape_PCA.ipynb`](examples/02_full_shape_PCA.ipynb)
- Apply PCA to analyze movement patterns
- Interpret principal components
- Visualise results and movement variations
- Compare experimental conditions

### TBC

## License

Distributed under the terms of the [MIT license](LICENSE).
