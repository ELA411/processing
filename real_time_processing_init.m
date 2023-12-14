% Author: Carl Larsson
%

function [n_eeg, d_eeg, notchFilt_50_eeg, notchFilt_100_eeg, n_emg, d_emg, notchFilt_50_emg, notchFilt_100_emg, notchFilt_150_emg] = real_time_processing_init(eeg_fs,emg_fs)

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

%--------------------------------------------------------------------------------------------------------
% EMG

% 20–500Hz fourth-order Butterworth bandpass filter.
[n_emg,d_emg] = butter(4,[20 499]/(emg_fs/2),"bandpass");

% 4th order IIR notch filter with quality factor 30 and 1 dB passband ripple
fo = 4;     % Filter order.
cf = 50/(emg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
% 50 Hz
notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * 1,qf,pbr);
notchFilt_50_emg = design(notchSpecs,'IIR','SystemObject',true);
% 100 Hz
notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * 2,qf,pbr);
notchFilt_100_emg = design(notchSpecs,'IIR','SystemObject',true);
% 150 Hz
notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * 3,qf,pbr);
notchFilt_150_emg = design(notchSpecs,'IIR','SystemObject',true);

%--------------------------------------------------------------------------------------------------------

end