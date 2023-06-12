# NF1_SchwannCell_data_analysis
## Steps to Reproduce
### Step 1. Clone the analysis repo
```sh
git clone https://github.com/WayScience/NF1_SchwannCell_data_analysis.git
```

### Step 2. Change the current path to the repo path

### Step 3. Populate the nf1_data_repo folder from the NF1_SchwannCell_data repo
```sh
git submodule update --init --recursive
```

### Step 4. Create the conda environment
```sh
conda env create analysis_env.yml
```
