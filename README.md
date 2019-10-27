# Pytorch-Lite
Easy to use small framework for faster model development and visualization 


## Usage
 - git pull 
 - create conda environment (conda env create -f env_specs.yml)
 - choose or define a model (`models/your_model.py`)
 - choose or define a dataset_loader
 - modify `constants.py`
 - run model 
 - run tensorboard (tensorboard --logdir=runs) to see the results

### Future works

 1. Refactor and improve backbone functionality
 2. Add examples and improve documentation
 3. Add multiple models and layers
 5. *Add more image enhancement techniques*
 6. *Add or facilitate more complex image visualization methods (matplotlib ... facets)*
 7. *Add python "artisan" for easier use* 