# Attention as Message Passing (ampnet)
Graph Neural Network architecture utilizing Multi-head attention during message passing
steps.


## Quickstart:
To set up the development environment, follow these steps:
1. Create an Anaconda environment with Python 3.7
2. Install Pytorch 1.9.0 through anaconda
3. Install PyTorch Geometric with the command:
    ```conda install -c pyg pyg```
4. Uninstall torch-spline-conv:
    ```pip uninstall torch-spline-conv```
5. Run experiments/cora_benchmark_graphsaint.py 

and then set the PYTHONPATH environment variable to the base directory of the repository (important for imports to work correctly):
```
export PYTHONPATH="/path/to/base/ampnet/directory" 
```

You will need to do this every time you open a new terminal window. To permanently set the environment variable, you can edit the configuration file of the shell you are using. For MacOS, you can do the following:
For zsh:
```
echo 'export PYTHONPATH="/path/to/base/ampnet/directory' >> ~/.zshenv
```

For bash:
echo 'export PYTHONPATH="/path/to/base/ampnet/directory' >> ~/.bash_profile


Current development is mostly occurring in experiments/cora_benchmark_graphsaint.py.
