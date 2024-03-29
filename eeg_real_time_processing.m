% Author: Carl Larsson
% Description: performs all processing of EEG data aswell as classifying the data
% Use: run when 1 window (250ms) of EEG is available.
% eeg_real_time_processing_init needs to be run before calling this function!
%========================================================================================================
% Copyright (c) 2023 Carl Larsson
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%========================================================================================================
% Inputs
% eeg_data: 1 window (250ms) of EEG data (format: matrix SxC, where S is samples and C is channels)
% eeg_fs: EEG sampling frequency in Hz
% window_size: EEG window size in s
% overlap: EEG window overlap in s
% W: the calculated CSP W matrix
% eeg_classifier: the trained EEG classifier (cross validated model)
% n_eeg, d_eeg: EEG butterworth highpass filter parameters
% notchFilt_50_eeg: EEG 50Hz notch filter
% notchFilt_100_eeg: EEG 100Hz notch filter
%========================================================================================================
% Outputs
% eeg_label: the classification label of the EEG data/window
%========================================================================================================
% Dependencies
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
% Wavelet Toolbox: https://se.mathworks.com/products/wavelet.html?s_tid=FX_PR_info
% FastICA: https://research.ics.aalto.fi/ica/fastica/
% wICA(data,varargin): https://se.mathworks.com/matlabcentral/fileexchange/55413-wica-data-varargin
% Common Spatial Patterns (CSP): https://se.mathworks.com/matlabcentral/fileexchange/72204-common-spatial-patterns-csp
% Statistics and Machine Learning Toolbox: https://se.mathworks.com/products/statistics.html
%========================================================================================================

function [eeg_label] = eeg_real_time_processing(eeg_data, eeg_fs, window_size, overlap, W, eeg_classifier, n_eeg, d_eeg, notchFilt_50_eeg, notchFilt_100_eeg)

%--------------------------------------------------------------------------------------------------------
% EEG Preprocessing
% Remove baseline wandering and DC offset, also the bandpass is necessary
% since log bandpower assumes signal has been bandpass filtered
eeg_data = filter(n_eeg,d_eeg,eeg_data);

% Removal of 50Hz noise and all of it's harmonics up to 100Hz.
eeg_data = notchFilt_50_eeg(eeg_data);
eeg_data = notchFilt_100_eeg(eeg_data);

% !!! CRASHES THE PROGRAM DURING ONLINE TRIALS !!!
%{
% Remove artifacts from EEG using wavelet enhanced ICA, W-ICA
% add 'verbose', 'off' in fastica
[wIC,A,~,~] = wICA(transpose(eeg_data));
% Artifacts
artifacts = transpose(A*wIC);
% Subtract artifacts from original signal to get "artifact free" signal
eeg_data = eeg_data - artifacts;
%}
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EEG Feature Extraction

% CSP filter data
eeg_data = transpose(W'*transpose(eeg_data));

eeg_features = zeros(1, 8); % Create matrix containing all extracted features beforehand

eeg_features(1, 1:4) = log(bandpower(eeg_data,eeg_fs,[0 eeg_fs/2])); % Log band power

% Extract variance of PSD of beta band from each window
for channel=1:4
    % Compute PSD using pwelch
    [psd, freq] = pwelch(eeg_data(:,channel), window_size*eeg_fs, overlap*eeg_fs, [], eeg_fs);
    
    % Extract power within the beta band
    beta_band = [12 30];
    % Extract power in beta band
    beta_idx = find(freq >= beta_band(1) & freq <= beta_band(2));
    eeg_features(4+channel) = var(psd(beta_idx));
end
%--------------------------------------------------------------------------------------------------------

%--------------------------------------------------------------------------------------------------------
% EEG Classification
eeg_label = predict(eeg_classifier.Trained{1}, eeg_features);
%--------------------------------------------------------------------------------------------------------

end