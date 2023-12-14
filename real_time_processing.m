% Author: Carl Larsson
%

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
eeg_data = log(var(eeg_data));
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EMG Feature Extraction
f_mav = jfemg('mav', emg_data); % Mean absolut value
f_wl = jfemg('wl', emg_data); % Waveform length
f_zc = jfemg('zc', emg_data); % Zero crossing
f_ssc = jfemg('ssc', emg_data); % Slope sign change
opts.order = 1; % Defines output dimension
f_ar = jfemg('ar', emg_data, opts); % Auto regressive

emg_data = [f_mav, f_wl, f_zc, f_ssc, f_ar];
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