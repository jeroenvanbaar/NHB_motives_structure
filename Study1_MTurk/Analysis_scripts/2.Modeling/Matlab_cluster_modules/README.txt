A quick readme on the modeling code.

First, this part of the code was run in Matlab on a Linux cluster. Hence there are .sh script files that submit many parallel jobs to this cluster, e.g. fitting a model to a participant's data multiple times with different starting parameters for the gradient descent algorithm. This also means that this code is not well integrated with the rest of the repo -- for instance, some data files are repeated here.

Basically there are four parts to the modeling code:
- "FeatureRL" folder contains all code to run and fit the different models from the paper to the participant data;
- "FeatureRL/Random_BS" folder contains the code to run the model with random basis sets (pseudo-motive sets) as described in Supplemental Results 1;
- Some jupyter notebooks (python) in the parent directory collect the resulting model fits for further analysis (e.g. model comparison);
- "Model_recovery" folder contains all code to run the model recovery in Supplemental Results 2.
- "FeatureRL/PPCs" folder contains code to produce posterior predictive checks, i.e. detailed plots of the behavior observed in the experiments and the behavior produced by the fitted model, side by side. Relevant for Figures 3C and S3D.

Enjoy!
