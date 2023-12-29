% Author: Carl Larsson
% Description: Creates all filters necessary for real time processing of EEG
% Use: should be run before "real time loop" to enable the use of the
% function eeg_real_time_processing
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
% eeg_fs: the sampling frequency of the EEG signal in Hz
%========================================================================================================
% Outputs
% n_eeg, d_eeg: EEG butterworth highpass filter parameters
% notchFilt_50_eeg: EEG 50Hz notch filter
% notchFilt_100_eeg: EEG 100Hz notch filter
%========================================================================================================
% Dependencies
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
%========================================================================================================

function [n_eeg, d_eeg, notchFilt_50_eeg, notchFilt_100_eeg] = eeg_real_time_processing_init(eeg_fs)

%--------------------------------------------------------------------------------------------------------
% EEG

% 4th order Butterworth bandpass filter, necessary since log bandpower assumes signal has been bandpass filtered.
[n_eeg,d_eeg] = butter(4,[0.1 99]/(eeg_fs/2),"bandpass");

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