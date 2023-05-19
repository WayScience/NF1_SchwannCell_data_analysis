# NF1_SchwannCell_data_analysis
## Steps to Reproduce
### Step 1. Clone the analysis repo
```sh
git clone https://github.com/WayScience/NF1_SchwannCell_data_analysis.git`<br>
```

### Step 2. cd into the repo<br>

### Step 3. Populate the nf1_data_repo folder from the NF1_SchwannCell_data repo
```sh
git submodule update --init --recursive
```

### Step 4.
```sh
# Create the Conda environment
conda env create analysis_env.yml
```

### Step 5. Open the code in your preferred IDE
