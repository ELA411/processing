% Author: Carl Larsson
% Description: performs all processing of EEG and EMG data aswell as classifying the data
% Use: run when 1 window (250ms) of EEG and EMG data is available.
% real_time_processing_init needs to be run before calling this function
%========================================================================================================
% Inputs
% eeg_data: 1 window (250ms) of EEG data (format: matrix SxC, where S is samples and C is channels)
% emg_data: 1 window (250ms) of EMG data (format: matrix SxC, where S is samples and C is channels)
% W: the calculated CSP W matrix
% eeg_classifier: the trained EEG classifier (cross validated model)
% emg_classifier: the trained EMG classifier (cross validated model)
% n_eeg, d_eeg: EEG butterworth highpass filter parameters
% notchFilt_50_eeg: EEG 50Hz notch filter
% notchFilt_100_eeg: EEG 100Hz notch filter
% n_emg, d_emg: EMG bandpass filter parameters
% notchFilt_50_emg: EMG 50Hz notch filter
% notchFilt_100_emg: EMG 100Hz notch filter
% notchFilt_150_emg: EMG 150Hz notch filter
%========================================================================================================
% Outputs
% eeg_label: the classification label of the EEG data/window
% emg_label: the classification label of the EMG data/window
%========================================================================================================
% Dependencies
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
% Wavelet Toolbox: https://se.mathworks.com/products/wavelet.html?s_tid=FX_PR_info
% FastICA: https://research.ics.aalto.fi/ica/fastica/
% wICA(data,varargin): https://se.mathworks.com/matlabcentral/fileexchange/55413-wica-data-varargin
% Common Spatial Patterns (CSP): https://se.mathworks.com/matlabcentral/fileexchange/72204-common-spatial-patterns-csp
% EMG Feature Extraction Toolbox: https://se.mathworks.com/matlabcentral/fileexchange/71514-emg-feature-extraction-toolbox
% Statistics and Machine Learning Toolbox: https://se.mathworks.com/products/statistics.html
%========================================================================================================

function [eeg_label,emg_label] = real_time_processing(eeg_data, emg_data, W, eeg_classifier, emg_classifier, n_eeg, d_eeg, notchFilt_50_eeg, notchFilt_100_eeg, n_emg, d_emg, notchFilt_50_emg, notchFilt_100_emg, notchFilt_150_emg)

%--------------------------------------------------------------------------------------------------------
% EEG Preprocessing
% Remove baseline wandering and DC offset
eeg_data = filter(n_eeg,d_eeg,eeg_data);

% Removal of 50Hz noise and all of it's harmonics up to 100Hz.
eeg_data = notchFilt_50_eeg(eeg_data);
eeg_data = notchFilt_100_eeg(eeg_data);

% Remove artifacts from EEG using wavelet enhanced ICA, W-ICA
% add 'verbose', 'off' in fastica
[wIC,A,~,~] = wICA(transpose(eeg_data));
% Artifacts
artifacts = transpose(A*wIC);
% Subtract artifacts from original signal to get "artifact free" signal
eeg_data = eeg_data - artifacts;

% CSP filter data
eeg_data = transpose(W'*transpose(eeg_data));
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EMG Preprocessing
% Removal of the 0Hz(the DC offset) and high frequency noise.
emg_data = filter(n_emg,d_emg,emg_data);

% Removal of 50Hz noise and all of it's harmonics up to 150Hz. 
emg_data = notchFilt_50_emg(emg_data);
emg_data = notchFilt_100_emg(emg_data);
emg_data = notchFilt_150_emg(emg_data);
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EEG Feature Extraction
eeg_data = log(var(eeg_data)); % Log variance
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EMG Feature Extraction
% Channel 1
f_mav_1 = jfemg('mav', emg_data(:,1)); % Mean absolut value
f_wl_1 = jfemg('wl', emg_data(:,1)); % Waveform length
f_zc_1 = jfemg('zc', emg_data(:,1)); % Zero crossing
f_ssc_1 = jfemg('ssc', emg_data(:,1)); % Slope sign change
opts.order = 1; % Defines output dimension
f_ar_1 = jfemg('ar', emg_data(:,1), opts); % Auto regressive

% Channel 2
f_mav_2 = jfemg('mav', emg_data(:,2)); % Mean absolut value
f_wl_2 = jfemg('wl', emg_data(:,2)); % Waveform length
f_zc_2 = jfemg('zc', emg_data(:,2)); % Zero crossing
f_ssc_2 = jfemg('ssc', emg_data(:,2)); % Slope sign change
opts.order = 1; % Defines output dimension
f_ar_2 = jfemg('ar', emg_data(:,2), opts); % Auto regressive

emg_data = [f_mav_1, f_wl_1, f_zc_1, f_ssc_1, f_ar_1, f_mav_2, f_wl_2, f_zc_2, f_ssc_2, f_ar_2];
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EMG Classification
eeg_label = predict(eeg_classifier.Trained{1}, eeg_data);
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EMG Classification
emg_label = predict(emg_classifier.Trained{1}, emg_data);
%--------------------------------------------------------------------------------------------------------

end