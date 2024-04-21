# Continuous-Time-GP-Dynamics
PyTorch Implementation for exact inference for continuous-time GP dynamics. 
This is part of the publication "Exact Inference for Continuous-Time Gaussian Process Dynamics" (AAAI 2024)

# Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication "Exact Inference for Continuous-Time Gaussian Process Dynamics" (AAAI 2024).
It will neither be maintained nor monitored in any way. 

# Installation
The required packages are listed in requirements.yml

# License
Continuous-time GP dynamics is open-sourced under the AGPL-3.0 license.
See the [LICENSE](LICENSE) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

# Getting Started
The main interaction with the framework is via command line:

## Command-line
In the following, we describe how to run the code.  

### Training script
A basic version of the algorithm can be run with default settings via command line

```bash
python main_local.py
```

To execute a specific experiment, a config file or multiple config files have to be provided via

```bash
python main_local.py --config_files <configname1>.json <configname2>.json
```

where the .json files will be taken from the folder /data/experiment_configs.

A separate training will be carried on for every random 
seed provided in the list.
The results will be stored in a separate folder with the number of
the seed, in the directory of the experiment in \run_data.

### Prediction script

After training a model, predictions are done on a separate script via

```bash
python predictions_local.py --experiment_dir <experiment_dir_1> <experiment_dir_2> ... --output_dir <output path>
```

where all the directories are intended as absolute paths.


To launch the prediction script for a single seed from command line, use

```bash
python prediction_generator.py --experiment_dir <experiment_dir_1> --save_dir <path where to store the results> --seed <seed1> <seed2> ...
```

### Metrics script

Seed-level metrics are evaluated in this script.

```bash
python metrics_generator.py --data_path <batch_directory> 
```

where <batch_directory> is the absolute path to a directory containing all the
experiments folders.

## Setting a .json Config-file

The two models available are the
[MultistepGP](continuous_time_gp/models/multistepgp.py) and the
[TaylorGP](continuous_time_gp/models/taylorgp.py).
Config files are available on the folder
[/data/experiment_configs](continuous_time_gp/data/experiment_configs)

### Multistep GP

The model [MultistepGP](continuous_time_gp/models/multistepgp.py) can be used
to predict trajectories by using an arbitrary multistep method, just by
providing the coefficients to the MultistepKernel model.
Since the coefficients of a generic multistep method depends on the
irregularity of the timeline, it is necessary to write a coefficient generator
that, given a timeline, returns the coefficients for the scheme as
torch.Tensors.

A coefficient generator is a class that inherits from NumericalScheme, it
should realize a class variable k with the order of the method and concretize
the method build_coeffs that generate the coefficients given a timeline:

```python
class backward_euler(NumericalScheme):

    k: int = 1

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
```

Here, timeline is a 1D torch.Tensor, and the return values are one 2D tensor
for alpha coefficients, one for beta coefficients, and one integer indicating
the order of the numerical method:

```python
return alpha, beta
```

To indicate an implemented numerical method in the config files, just write
the name of the function on the string data_config.integration_rule:

```json
"data_config": {
        "method": "multistep",
        "integration_rule": "bdf_2",
        ...
}
```

Further, the GP has to be adapted to a MultistepGP with a multistep kernel via

```json
        "gp_config": {
            "MultistepGPConfig": "",
            "kernel_config": {
                "MultistepKernelConfig": "",
				...
                }
            }
```

An implemented numerical method can be used both for training but also for
prediction, and it is possible to train with a numerical scheme and
to predict with a different one just by indicating the different prediction
integration rule in prediction_config.integrator_config.integration_rule:

```json
"prediction_config": {
       ...
       "integrator_config": {
           "integration_rule": "bdf_2",
           "MultistepIntegratorConfig": ""
       }
   }
```

Additionally, since coefficients are not constant when the timelines are
irregular, the classes ab_k, am_k, bdf_k able to dynamically generate the
coefficients are provided.
Prediction with an adaptive step-size Runge-Kutta method (RK45) are controlled by 
the field TrajectorySamplerConfig:

```json
"prediction_config": {
      "integrator_config": 
            {
                "TrajectorySamplerConfig": ""
            }
   }
```
### Taylor GP

[TaylorGP](continuous_time_gp/models/taylorgp.py) model that uses Taylor series
formula both for training and prediction. We provide the options TaylorGP and ExactTaylorGP controlled in the field gp_config. 
A TaylorGP approximates every higher order derivative with an independent GP and thus allows for parallelized training.
An ExactTaylorGP applies kernels and hyperparameters that are adapted between higher-order derivatives. 


For an ExactTaylorGP or TaylorGP, only the order of the Taylor expansion has to be provided via the integer "order".
As before, the usage of a TaylorGP is indicated in the data_config via 

```json
"data_config": {
        "method": "taylor",
        "integration_rule": "",
		...	
            }
```

Further, it has to be adapted in the gp_config. To train a TaylorGP with independent GPs, use 

```json
"gp_config": {
	 "TaylorGPConfig": "",
	 ...
	}
```

To train an ExactTaylorGP with adapted GPs, use

```json
"gp_config": {
	 "ExactTaylorGPConfig": "",
	 ...
	}
```

### Kernel

The code provide basic kernels in the
[kernels.basekernel](continuous_time_gp/models/kernels/basekernel.py) module
like Squared Exponential Kernels, and also an abstract class
[kernels.customkernel](continuous_time_gp/models/kernels/customkernel.py)
that can be used for more complex kernels involving also the composition of
other basic kernels (i.e. a linear combination of Squared Exponential
Kernels).

To configure a [BaseKernel](continuous_time_gp/models/kernels/basekernel.py),
the settings in the json for a SE Kernel are:

```json
"basekernel_config": {
                   "num_fourier_feat": 256,
                   "act_func": "log",
                   "noise_par": 0.5,
                   "lengthscale": [
                     [[1.0, 1.0]],
                     [[1.0, 1.0]]
                     ],
                   "sigma_k": [0.5, 0.5],
                   "ard": true
}
```

Where each element of the list of lengthscale and sigma_k are the set of
lengthscales and kernel std. for all the different GPs. To note that a Taylor
GP can contain multiple RBF kernels, reason why the lengthscale fields is a
List[List[List[]]].
### Train and Evaluation data

To configure a simulated experiment, its initial conditions, timeline
etc., the config file to be modified is

```json
"simulation_config": {
           "dt": 0.14,
           "t_endpoints": [
               0.0,
               7.0
           ],
           "y0": [
               -1.5,
               2.5
           ],
           "bound": 0.0,
           "noise_std": 0.05
}
```

An identical configuration for the test_config field will generate
the prediction trajectory. 

The "bound" field controls the irregularity of timeline samples, noise_std determines the standard deviation of
random gaussian noise added to the trajectories.

To implement a new model, add a function in the
[data.dynamical_models.py](continuous_time_gp/data/dynamical_models.py)
module: The data will be constructed by integrating with a higher order
Runge-Kutta integration methods coming from the scipy library:

```python
def damped_harmonic_oscillator(t, z, kwargs=None)
```

With t as first component and z as second component (z can be
multidimensional).

A full example illustrating also how the return parameters should be put in a
list:

```python
def damped_harmonic_oscillator(t, z, kwargs=None):
    """ODE for the Damped Harmonic-Oscillator model."""

    x, y = z
    return [- 0.1*x**3 + 2.0 * y**3, - 2*x**3 - 0.1 * y**3]
```

For the real world datasets (as the MoCap data) these steps are not necessary and will be bypassed. 
### Prediction
Prediction of new trajectories can be done both by using the posterior mean formulas, or via decoupled sampling [1]. 
The prediction scheme is indicated in the config file via

```json
"prediction_config": {
       "ds_pred": true,
       "mean_pred": true,
       "ds_prior_noise": true,
       "ds_num_trajectories": 50,
       "integrator_config": [
            {
                "integration_rule": "am_2",
                "MultistepIntegratorConfig": ""
            },
            {
                "TrajectorySamplerConfig": ""
            }
        ]
   }
```

If both are selected, both methods will be used. For decoupled sampling,
<ds_num_trajectories> trajectories will be simulated and the statistical mean and variance are calculated. 

The integrator_config field is a list of IntegratorConfig objects: Predictions
will be repeated for every single one of those integrators.

### Noise 
This framework provides the possibility to include different types of noise, controlled in the config file via the field "noise_mode". 
Especially, we provide the noise mode "exact_correlated". Here it is assumed that the noise on the trajectory is i.i.d. normal distributed noise. 
Depending on the integrator, this yields correlated noise between transformed observations.
The noise mode "exact" considers a diagonal matrix on the transformed data ignoring correlations.
This is an approximation but can lead to faster computations and more stable computations. 
The noise modes "direct" considers a diagonal noise matrix with identical entries. The mode "noiseless" does not consider noise.  
For mathematical details, we refer to the paper. 

# Note
This implementation is written by Nicholas Tagliapietra and Katharina Ensinger 
 
#References
[1] J. Wilson et al. Efficiently sampling functions from Gaussian process posteriors. In Internation Conference on Machine Learning (2020).