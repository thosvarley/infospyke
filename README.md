# infospyke
Information theoretic analysis of binary spiking data, aimed primarilly at researchers interetested in constructing networks based on bivariate information measures (mutual information, transfer entropy) 

Infospyke is designed for computational neuroscientists working with spiking neural data represented in a binary format (multi-electrode array spiking data, CA imaging, binarized fMRI data, etc). 
The code leverages the following assumptions for extremely rapid computation of information-theoretic measues:
1) The data is binary (spike = 1, quiescent = 0). 
2) For every channel, the number of samples is the same. 

Multi-dimensional spiking time-series are represented in a sparse, dictionary-based format:

```
sparse = {nbins : number_of_bins,

            channels :  {0 : set({timestamp_01, timestamp_02, timestamp_03...})}
            
                        {1 : set({timestamp_11, timestamp_12, timestamp_13...})}
          }
```
 
This sparse, dictionary-based format means that the computational complexity grows with the number of spikes, as opposed to the number of bins, allowing for rapid anlaysis of large datasets. 

## Installation
In the downloaded directory, run ``python setup.py install``
This will compile the Cython code. 
From there you can ``import`` like any other python code. 

## Functions

Currently implemented analyses are:

Shannon entropy (base 2)
Joint entropy (bivariate)
Conditional Entropy (bivariate)
Mutual Information (bivariate)
Conditional Mutual Information (trivariate)
Transfer Entropy (bivariate, variable lags). 

At some point we may implement basic partial-information decomposition (PID). 

We will also soon be including measures of integration. 

