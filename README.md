# Processing
Open source code for processing and training of classifier for a hybrid BCI system using EEG and EMG.

## Built with
+ Matlab R2023b
+ Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
+ Wavelet Toolbox: https://se.mathworks.com/products/wavelet.html?s_tid=FX_PR_info
+ FastICA: https://research.ics.aalto.fi/ica/fastica/
+ wICA(data,varargin): https://se.mathworks.com/matlabcentral/fileexchange/55413-wica-data-varargin
+ Common Spatial Patterns (CSP): https://se.mathworks.com/matlabcentral/fileexchange/72204-common-spatial-patterns-csp
+ EMG Feature Extraction Toolbox: https://se.mathworks.com/matlabcentral/fileexchange/71514-emg-feature-extraction-toolbox
+ Synthetic Minority Over-sampling Technique (SMOTE): https://se.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote
+ Statistics and Machine Learning Toolbox: https://se.mathworks.com/products/statistics.html

## Usage
For offline processing, training and validation of a classifier use the following code:
```
training_classifiers.m
```
Note that the following parts of the code needs to be altered: alter the search paths and files to the desired paths and files (for data sets) in section "Load data", change eeg_fs and emg_fs in the "Load data" section to the sample rate of the EEG and EMG.


The following code and functions are for online use.
The following function needs to be called before using the function for EEG online processing is called:
```
eeg_real_time_processing_init.m
```

Call this function to perform all EEG online processing and classification on one window of EEG data.
```
eeg_real_time_processing.m
```

The following function needs to be called before using the function for EMG online processing is called:
```
emg_real_time_processing_init.m
```

Call this function to perform all EEG online processing and classification on one window of EMG data.
```
emg_real_time_processing.m
```

## Disclaimer
The w-ICA part of the code is commented out because it causes a crash during online trials after a random amount of time.

## Contact
Carl Larsson - cln20001@student.mdu.se
