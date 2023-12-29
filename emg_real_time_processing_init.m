% Author: Carl Larsson
% Description: Creates all filters necessary for real time processing of EMG
% Use: should be run before "real time loop" to enable the use of the
% function emg_real_time_processing
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
% emg_fs: the sampling frequency of the EMG signal in Hz
%========================================================================================================
% Outputs
% n_emg, d_emg: EMG bandpass filter parameters
% notchFilt_50_emg: EMG 50Hz notch filter
% notchFilt_100_emg: EMG 100Hz notch filter
% notchFilt_150_emg: EMG 150Hz notch filter
%========================================================================================================
% Dependencies
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
%========================================================================================================

function [n_emg, d_emg, notchFilt_50_emg, notchFilt_100_emg, notchFilt_150_emg] = emg_real_time_processing_init(emg_fs)

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