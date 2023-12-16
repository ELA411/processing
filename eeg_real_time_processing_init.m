% Author: Carl Larsson
% Description: Creates all filters necessary for real time processing of EEG
% Use: should be run before "real time loop" to enable the use of the
% function real_time_processing
%========================================================================================================
% Inputs
% eeg_fs: the sampling frequency of the EEG signal in Hz
% emg_fs: the sampling frequency of the EMG signal in Hz
%========================================================================================================
% Outputs
% n_eeg, d_eeg: EEG butterworth highpass filter parameters
% notchFilt_50_eeg: EEG 50Hz notch filter
% notchFilt_100_eeg: EEG 100Hz notch filter
% n_emg, d_emg: EMG bandpass filter parameters
% notchFilt_50_emg: EMG 50Hz notch filter
% notchFilt_100_emg: EMG 100Hz notch filter
% notchFilt_150_emg: EMG 150Hz notch filter
%========================================================================================================
% Dependencies
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
%========================================================================================================

function [n_eeg, d_eeg, notchFilt_50_eeg, notchFilt_100_eeg] = eeg_real_time_processing_init(eeg_fs)

%--------------------------------------------------------------------------------------------------------
% EEG

% 4th order Butterworth highpass filter 0.1hz cut off frequency.
[n_eeg,d_eeg] = butter(4,(0.1)/(eeg_fs/2),"high");

% 4th order IIR notch filter with quality factor 30 and 1 dB passband ripple
fo = 4;     % Filter order.
cf = 50/(eeg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
% 50 Hz
notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * 1,qf,pbr);
notchFilt_50_eeg = design(notchSpecs,'IIR','SystemObject',true);
% 100 Hz
notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * 2,qf,pbr);
notchFilt_100_eeg = design(notchSpecs,'IIR','SystemObject',true);

%--------------------------------------------------------------------------------------------------------

end