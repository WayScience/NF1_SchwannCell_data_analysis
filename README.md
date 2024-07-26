# NF1_SchwannCell_data_analysis
## Repository Structure
Here we analyze NF1 morphology data from the [NF1 Cell painting Data](https://github.com/WayScience/nf1_cellpainting_data) repo. Our analysis is categorized as follows:

| Module | Purpose | Description |
| :---- | :----- | :---------- |
| [0.data_analysis](./0.data_analysis/) | Analyze NF1 Data | We find interesting patterns in the NF1 cell morphology data. |
| [1.train_models](./1.train_models/) | Train NF1 Models | Optimize the NF1 model by training multiple models with a random search. |
| [2.evaluate_models](./2.evaluate_models/) | Evaluate Final NF1 Model | After training the final NF1 model we evaluate model performance and model feature importances. |
| [3.figures](./3.figures/) | Visualize Analysis Results| We then interpret our results visually. |

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
