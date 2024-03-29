%% Code for training EEG and EMG classifiers
%========================================================================================================
% Author: Carl Larsson
% Description: Performs all processing of EEG and EMG data including
% training of classifier and evaluation of trained model
% Use: Alter the search paths and files to the desired paths and files (for data sets) in section "Load data", change eeg_fs and
% emg_fs in the "Load data" section to the sample rate of the EEG and EMG
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
% Dependencies
% Matlab R2023b or later
% Signal Processing Toolbox: https://se.mathworks.com/products/signal.html
% Wavelet Toolbox: https://se.mathworks.com/products/wavelet.html?s_tid=FX_PR_info
% FastICA: https://research.ics.aalto.fi/ica/fastica/
% wICA(data,varargin): https://se.mathworks.com/matlabcentral/fileexchange/55413-wica-data-varargin
% Common Spatial Patterns (CSP): https://se.mathworks.com/matlabcentral/fileexchange/72204-common-spatial-patterns-csp
% EMG Feature Extraction Toolbox: https://se.mathworks.com/matlabcentral/fileexchange/71514-emg-feature-extraction-toolbox
% Synthetic Minority Over-sampling Technique (SMOTE): https://se.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote
% Statistics and Machine Learning Toolbox: https://se.mathworks.com/products/statistics.html
%========================================================================================================
%% Clear variables and close figures
clear all
close all
%% Load data
raw_eeg_data = load("data_sets\EEG_Pontus-Left_side_cluster_50_reps_2024-01-05_181626.txt"); % EEG data set (expected column format: channels, labels, package ID, timestamp. Each row is expected to be subsequent observations)
eeg_fs = 200; % EEG sample rate
raw_emg_data = load("data_sets\EMG_Pontus-Ch_1_longitude_Ch_2_transverse_2024-01-05_180125.txt"); % EMG data set (expected column format: channels, labels, package ID, timestamp. Each row is expected to be subsequent observations)
emg_fs = 1000; % EMG sample rate
%% Display data lost
%------------------------------------------------------------------------------------------------
% EEG

fprintf("=======================================================================================\n")
data_lost = 0;
last_last_id = raw_eeg_data(1, 6);
last_id = raw_eeg_data(2, 6);
for package = 3:length(raw_eeg_data)
    % EEG samples are sent two at a time with the same package ID (diplicates)
    if (raw_eeg_data(package, 6) ~= last_id) && (last_id ~= last_last_id) % Duplicate half lost
        data_lost = data_lost + 1;
    elseif raw_eeg_data(package, 6) == last_id % Duplicate
    % EEG package ID goes from 100 to 200
    elseif raw_eeg_data(package, 6) == 100 + mod((last_id + 1) - 100, 100) % Next package ID is previous + 1
    else
        data_lost = data_lost + 2; % Package ID skipped, a duplicate pair is lost
    end
    last_last_id = last_id;
    last_id = raw_eeg_data(package, 6);
end
disp(['EEG Data lost: ', num2str(data_lost)]);
%------------------------------------------------------------------------------------------------
% EMG

data_lost = 0;
for package = 2:length(raw_emg_data)
    % If next package doesn't have last package ID + 1, then a package has been lost
    % EMG package ID is mod 1000
    if raw_emg_data(package, 4) ~= mod(raw_emg_data(package-1, 4) + 1, 1000)
        data_lost = data_lost + 1;
    end
end
disp(['EMG Data lost: ', num2str(data_lost)]);
%% Display data quality 
%------------------------------------------------------------------------------------------------
% EEG

fprintf("=======================================================================================\n")
% Extract signal-to-noise (SNR), signal to noise and distortion ratio (SINAD), total harmonic distortion (THD).
sFE = signalTimeFeatureExtractor(SampleRate=eeg_fs, SNR = true, SINAD = true, THD = true);
fprintf("EEG data quality")
eeg_quality = extract(sFE,raw_eeg_data(:, 1:4));
fprintf('\nChannel 1:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',eeg_quality(:,1,1),eeg_quality(:,2,1), eeg_quality(:,3,1));
fprintf('\nChannel 2:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',eeg_quality(:,1,2),eeg_quality(:,2,2), eeg_quality(:,3,2));
fprintf('\nChannel 3:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',eeg_quality(:,1,3),eeg_quality(:,2,3), eeg_quality(:,3,3));
fprintf('\nChannel 4:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',eeg_quality(:,1,4),eeg_quality(:,2,4), eeg_quality(:,3,4));

% Find period of samples for EEG
T_eeg = 0;
for package = 2:length(raw_eeg_data)
    T_eeg = T_eeg + (raw_eeg_data(package, end)-raw_eeg_data(package-1, end));
end
T_eeg = T_eeg/length(raw_eeg_data);
formatSpec = '%.2f';
disp(['EEG recorded sample rate: ', num2str(1/T_eeg, formatSpec), ' Hz']);
%------------------------------------------------------------------------------------------------
% EMG

fprintf("EMG data quality")
emg_quality = extract(sFE,raw_emg_data(:, 1:2));
fprintf('\nChannel 1:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',emg_quality(:,1,1),emg_quality(:,2,1), emg_quality(:,3,1));
fprintf('\nChannel 2:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',emg_quality(:,1,2),emg_quality(:,2,2), emg_quality(:,3,2));

% Find period of samples for EMG
T_emg = 0;
for package = 2:length(raw_emg_data)
    T_emg = T_emg + (raw_emg_data(package, end)-raw_emg_data(package-1, end));
end
T_emg = T_emg/length(raw_emg_data);
formatSpec = '%.2f';
disp(['EMG recorded sample rate: ', num2str(1/T_emg, formatSpec), ' Hz']);
%% Visualize signal
%------------------------------------------------------------------------------------------------
% EEG

% Plot all EEG channels
figure
EEG_plot_handle = tiledlayout(4,1);
nexttile;
plot(raw_eeg_data(:,1))
title('Channel 1')
nexttile;
plot(raw_eeg_data(:,2))
title('Channel 2')
nexttile;
plot(raw_eeg_data(:,3))
title('Channel 3')
nexttile;
plot(raw_eeg_data(:,4))
title('Channel 4')

% Add title etc
title(EEG_plot_handle,'Raw EEG data')
xlabel(EEG_plot_handle,'Time (s)')
ylabel(EEG_plot_handle,'Voltage (v)')
%------------------------------------------------------------------------------------------------
% EMG

% Plot all EMG channels
figure
EMG_plot_handle = tiledlayout(2,1);
nexttile;
plot(raw_emg_data(:,1))
title('Channel 1')
nexttile;
plot(raw_emg_data(:,2))
title('Channel 2')

% Add title etc
title(EMG_plot_handle,'Raw EMG data')
xlabel(EMG_plot_handle,'Time (s)')
ylabel(EMG_plot_handle,'Voltage (v)')
%% Preprocessing
%------------------------------------------------------------------------------------------------
% EEG

%------------------------------------------------------------------------------------------------
% Artifact removal: https://se.mathworks.com/matlabcentral/fileexchange/55413-wica-data-varargin 
% 
% Dependency: https://github.com/biotrump/RADICAL-matlab
% 
% Dependency: https://research.ics.aalto.fi/ica/fastica/
% 
% CSP filtering: https://se.mathworks.com/matlabcentral/fileexchange/72204-common-spatial-patterns-csp 
%------------------------------------------------------------------------------------------------

% Remove baseline wandering and DC offset
% 4th order Butterworth bandpass filter, necessary since log bandpower assumes the signal has been bandpass filtered.
[n,d] = butter(4,[0.1 99]/(eeg_fs/2),"bandpass");
filtered_eeg_data = filter(n,d,raw_eeg_data(:,1:4));

% Removal of 50Hz noise and all of it's harmonics up to 100Hz. 
% The noise is caused by magnetic fields generated by powerlines.
% Using a IIR notch filter of the fourth order as it's at that point the 50Hz noise peak on most signals practicaly becomes zero if a quality factor of about 25-35 is being used.
fo = 4;     % Filter order.
cf = 50/(eeg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
for harmonic = 1:2
    notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * harmonic,qf,pbr);
    notchFilt = design(notchSpecs,'IIR','SystemObject',true);
    filtered_eeg_data = notchFilt(filtered_eeg_data); 
end

%{
% Remove artifacts from EEG using wavelet enhanced ICA, W-ICA
% add 'verbose', 'off' in fastica
[wIC,A,~,~] = wICA(transpose(filtered_eeg_data));
% Artifacts
artifacts = transpose(A*wIC);
% Subtract artifacts from original signal to get "artifact free" signal
filtered_eeg_data = filtered_eeg_data - artifacts;
%}

% CSP
[row_size, ~] = size(filtered_eeg_data);
label_1 = find(raw_eeg_data(:,5) == 1); % Find all indicies that belong to class 1
class_0_data = filtered_eeg_data(~ismember(1:row_size,label_1), :); % All indicies except the ones that belong to class 1 thus belong to class 0
class_1_data = filtered_eeg_data(label_1, :); % Class 1
% Calculate and fins W matrix
[W,~,~] = csp(transpose(class_0_data),transpose(class_1_data));
% Save W matrix for real time
save("saved_variables\W_matrix.mat","W");
% CSP filtering is applied to each window individually later

% 1. 2. 3. 4. channel 1,2,3,4
% 5. labels
% 6. package ID
% 7. timestamp
filtered_eeg_data = [filtered_eeg_data raw_eeg_data(:,5:7)];

%------------------------------------------------------------------------------------------------

% Plot all EEG channels
figure
EEG_filtered_plot_handle = tiledlayout("vertical");
nexttile;
plot(raw_eeg_data(:,1), 'r--')
hold on
plot(filtered_eeg_data(:,1), 'g')
hold off
title('Channel 1')
legend('Raw','Filtered')
nexttile;
plot(raw_eeg_data(:,2), 'r--')
hold on
plot(filtered_eeg_data(:,2), 'g')
hold off
title('Channel 2')
legend('Raw','Filtered')
nexttile;
plot(raw_eeg_data(:,3), 'r--')
hold on
plot(filtered_eeg_data(:,3), 'g')
hold off
title('Channel 3')
legend('Raw','Filtered')
nexttile;
plot(raw_eeg_data(:,4), 'r--')
hold on
plot(filtered_eeg_data(:,4), 'g')
hold off
title('Channel 4')
legend('Raw','Filtered')

% Add title etc
title(EEG_filtered_plot_handle,'Raw vs Filtered EEG data')
xlabel(EEG_filtered_plot_handle,'Time (s)')
ylabel(EEG_filtered_plot_handle,'Voltage (v)')
%------------------------------------------------------------------------------------------------
% EMG

% Removal of the 0Hz(the DC offset) and high frequency noise.
% 20�500Hz fourth-order Butterworth bandpass filter.
[n,d] = butter(4,[20 499]/(emg_fs/2),"bandpass");
filtered_emg_data = filter(n,d,raw_emg_data(:,1:2));

% Removal of 50Hz noise and all of it's harmonics up to 150Hz. 
% The noise is caused by magnetic fields generated by powerlines.
% Using a IIR notch filter of the fourth order as it's at that point the 50Hz noise peak on most signals practicaly becomes zero if a quality factor of about 25-35 is being used.
fo = 4;     % Filter order.
cf = 50/(emg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
for harmonic = 1:3
    notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * harmonic,qf,pbr);
    notchFilt = design(notchSpecs,'IIR','SystemObject',true);
    filtered_emg_data = notchFilt(filtered_emg_data); 
end

% 1. 2. channel 1, 2
% 3. labels
% 4. package ID
% 5. time stamp
filtered_emg_data = [filtered_emg_data raw_emg_data(:,3:5)];

%------------------------------------------------------------------------------------------------

% Plot all EMG channels
figure
EMG_plot_handle = tiledlayout(2,1);
nexttile;
plot(raw_emg_data(:,1),'r--')
hold on
plot(filtered_emg_data(:,1),'g')
hold off
title('Channel 1')
legend('Raw','Filtered')
nexttile;
plot(raw_emg_data(:,2),'r--')
hold on
plot(filtered_emg_data(:,2),'g')
hold off
title('Channel 2')
legend('Raw','Filtered')

% Add title etc
title(EMG_plot_handle,'Raw vs Filtered EMG data')
xlabel(EMG_plot_handle,'Time (s)')
ylabel(EMG_plot_handle,'Voltage (v)')
%% Segmentation
%------------------------------------------------------------------------------------------------
% EEG

% Window EEG signal into 250ms windows with 50ms overlap
window_size = 0.250;                        % window size s
overlap = 0.050;                            % window overlap s
[eeg_1, ~] = buffer(filtered_eeg_data(:,1),window_size*eeg_fs, (overlap/2)*eeg_fs, 'nodelay'); % Channel 1
[eeg_2, ~] = buffer(filtered_eeg_data(:,2),window_size*eeg_fs, (overlap/2)*eeg_fs, 'nodelay'); % Channel 2
[eeg_3, ~] = buffer(filtered_eeg_data(:,3),window_size*eeg_fs, (overlap/2)*eeg_fs, 'nodelay'); % Channel 3
[eeg_4, ~] = buffer(filtered_eeg_data(:,4),window_size*eeg_fs, (overlap/2)*eeg_fs, 'nodelay'); % Channel 4
[eeg_label, ~] = buffer(filtered_eeg_data(:,5),window_size*eeg_fs, (overlap/2)*eeg_fs, 'nodelay'); % Labels
%------------------------------------------------------------------------------------------------
% EMG

% Window EMG signal into 250ms windows with 50ms overlap
window_size = 0.250;                        % window size s
overlap = 0.050;                            % window overlap s
[emg_1, ~] = buffer(filtered_emg_data(:,1),window_size*emg_fs, (overlap/2)*emg_fs, "nodelay"); % Channel 1
[emg_2, ~] = buffer(filtered_emg_data(:,2),window_size*emg_fs, (overlap/2)*emg_fs, "nodelay"); % Channel 2
[emg_label, ~] = buffer(filtered_emg_data(:,3),window_size*emg_fs, (overlap/2)*emg_fs, "nodelay"); % Labels
%% Feature extraction
%------------------------------------------------------------------------------------------------
% EEG

% Extract features from each window and channel
[~, col_size] = size(eeg_1);
psd_alpha = zeros(1, 4);
psd_beta = zeros(1, 4);
eeg_1_features = zeros(col_size, 8); % Create matrix containing all extracted features from each window beforehand
for window=1:col_size
    eeg_channels_window = [eeg_1(:,window), eeg_2(:,window), eeg_3(:,window), eeg_4(:,window)]; % Window with all channels
    csp_eeg = transpose(W'*transpose(eeg_channels_window)); % CSP filter window
    log_pow = log(bandpower(csp_eeg,eeg_fs,[0 eeg_fs/2])); % Log power

    % Extract variance of PSD of beta band from each window
    for channel=1:4
        % Compute PSD using pwelch
        [psd, freq] = pwelch(eeg_channels_window(:,channel), window_size*eeg_fs, overlap*eeg_fs, [], eeg_fs);
        
        % Extract power within the alpha and beta bands
        %alpha_band = [7, 13];
        beta_band = [12 30];
        % Extract power in alpha band
        %alpha_idx = find(freq >= alpha_band(1) & freq <= alpha_band(2));
        %psd_alpha(channel) = var(psd(alpha_idx));
        % Extract power in beta band
        beta_idx = find(freq >= beta_band(1) & freq <= beta_band(2));
        psd_beta(channel) = var(psd(beta_idx));
    end

    eeg_1_features(window,:) = [log_pow, psd_beta];
end

% Labels
% Labels for each window is calculated as the majority class of that window
[~, col_size] = size(eeg_label);
eeg_label_window = zeros(col_size, 1);
for window=1:col_size
    eeg_label_window(window,:) = round(mean(eeg_label(:,window)));
end

% 1:4 log band power of each channel
% 5:8 variance of PSD of beta band from each channel
% 9 labels
eeg_features = [eeg_1_features eeg_label_window];
%------------------------------------------------------------------------------------------------
% EMG

%------------------------------------------------------------------------------------------------
% Feature extraction: https://se.mathworks.com/matlabcentral/fileexchange/71514-emg-feature-extraction-toolbox
%------------------------------------------------------------------------------------------------

% Channel 1
% Extract features from each window
[~, col_size] = size(emg_1);
emg_1_features = zeros(col_size, 5); % Create matrix containing all extracted features from each window beforehand
for window=1:col_size
    f_mav = jfemg('mav', emg_1(:,window)); % Mean absolut value
    f_wl = jfemg('wl', emg_1(:,window)); % Waveform length
    f_zc = jfemg('zc', emg_1(:,window)); % Zero crossing
    f_ssc = jfemg('ssc', emg_1(:,window)); % Slope sign change
    opts.order = 1; % Defines output dimension
    f_ar = jfemg('ar', emg_1(:,window), opts); % Auto regressive

    emg_1_features(window,:) = [f_mav, f_wl, f_zc, f_ssc, f_ar];
end

[~, col_size] = size(emg_2);
emg_2_features = zeros(col_size, 5); % Create matrix containing all extracted features from each window beforehand
for window=1:col_size
    f_mav = jfemg('mav', emg_2(:,window)); % Mean absolut value
    f_wl = jfemg('wl', emg_2(:,window)); % Waveform length
    f_zc = jfemg('zc', emg_2(:,window)); % Zero crossing
    f_ssc = jfemg('ssc', emg_2(:,window)); % Slope sign change
    opts.order = 1; % Defines output dimension
    f_ar = jfemg('ar', emg_2(:,window), opts); % Auto regressive

    emg_2_features(window,:) = [f_mav, f_wl, f_zc, f_ssc, f_ar];
end

% Labels
% Labels for each window is calculated as the majority class of that window
[~, col_size] = size(emg_label);
emg_label_window = zeros(col_size, 1);
for window=1:col_size
    emg_label_window(window,:) = round(mean(emg_label(:,window)));
end

% 1:5 channel 1 features
% 6:10 channel 2 features
% 11 labels
emg_features = [emg_1_features emg_2_features emg_label_window];
%% Train classifier
%------------------------------------------------------------------------------------------------
% EEG

%------------------------------------------------------------------------------------------------
% Oversampling: https://se.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote
% 
% Permutation test: https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf
%------------------------------------------------------------------------------------------------

% Fix class imbalance with Synthetic Minority Over-sampling Technique (SMOTE)
[smote_data, smote_label, ~, ~] = smote(eeg_features(:, 1:end-1),[], 5, 'Class', eeg_features(:,end));
balanced_eeg_data = [smote_data smote_label];

% 80 train 20 test
cv = cvpartition(size(balanced_eeg_data,1),'HoldOut',0.2);
idx = cv.test;
balanced_eeg_data_train = balanced_eeg_data(~idx,:);
balanced_eeg_data_test = balanced_eeg_data(idx,:);

% Train classifier using 5 fold cross validation
% kernelscale has major impact and alters how the loss functions work
eeg_classifier = fitcsvm(balanced_eeg_data_train(:,1:end-1), balanced_eeg_data_train(:,end),"KernelFunction","rbf","CrossVal","on","KFold",5); % RBF SVM
save("trained_classifiers\eeg_classifier.mat","eeg_classifier")

fprintf("=======================================================================================\n")
% Accuracy
% Accuracy for each fold
formatSpec = '%.2f';
accuracy_eeg_fold = kfoldLoss(eeg_classifier, 'Mode', 'individual', 'LossFun', 'classiferror');
disp(['SVM 1st fold accuracy : ', num2str((1-accuracy_eeg_fold(1))*100, formatSpec), '%']);
disp(['SVM 2nd fold accuracy : ', num2str((1-accuracy_eeg_fold(2))*100, formatSpec), '%']);
disp(['SVM 3rd fold accuracy : ', num2str((1-accuracy_eeg_fold(3))*100, formatSpec), '%']);
disp(['SVM 4th fold accuracy : ', num2str((1-accuracy_eeg_fold(4))*100, formatSpec), '%']);
disp(['SVM 5th fold accuracy : ', num2str((1-accuracy_eeg_fold(5))*100, formatSpec), '%']);

% Average accuracy over all folds
disp(['SVM average accuracy : ', num2str((mean(1-accuracy_eeg_fold))*100, formatSpec), '% (+-', num2str(std(1-accuracy_eeg_fold)*100, formatSpec), '%)']);

% Confusion matrix
% Predict on test data
predicted = predict(eeg_classifier.Trained{1}, balanced_eeg_data_test(:,1:end-1));
eeg_confusion_matrix = confusionmat(balanced_eeg_data_test(:,end), predicted);
figure
confusionchart(eeg_confusion_matrix)

% Statistical tests
%------------------------------------------------------------------------------------------------
% Permutation test 1 (shuffle predicted labels)
true_labels = balanced_eeg_data_test(:,end);
predicted_labels = predicted;

% Calculate accuracy for original classifier
actual_accuracy = sum(true_labels == predicted_labels,'all')/numel(predicted_labels);

num_permutations = 100;
permuted_accuracy = zeros(1,num_permutations);
for k = 1:num_permutations
    % Shuffle predicted labels
    shuffled_labels = predicted_labels(randperm(length(predicted_labels)));

    % Calculate accuracy when predicted labels have been shuffled
    permuted_accuracy(k) = sum(true_labels == shuffled_labels,'all')/numel(shuffled_labels);
end

% P value is the fraction of times that the shuffled predicted labels
% performed better than actual predicted labels 
p_value = (sum(permuted_accuracy >= actual_accuracy) + 1) / (num_permutations + 1);
disp(['SVM permutation test 1 p-value :', num2str(p_value,2)])

if p_value < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier performance is better than random chance.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
% Permutation test 2 (shuffle true labels)
true_labels = balanced_eeg_data_train(:,end);
data_set = balanced_eeg_data_train(:,1:end-1);

% Calculate accuracy for original classifier
actual_accuracy = mean(1-accuracy_eeg_fold);

num_permutations = 100;
permuted_accuracy = zeros(1,num_permutations);
for k = 1:num_permutations
    % Shuffle the true labels and train classifier when labels no longer has a true connection to the data
    shuffled_true = true_labels(randperm(length(true_labels)));
    perm_classifier = fitcsvm(data_set, shuffled_true,"KernelFunction","rbf","CrossVal","on","KFold",5); % SVM

    % CV accuracy
    accuracy_shuffled = kfoldLoss(perm_classifier, 'Mode', 'individual', 'LossFun', 'classiferror');

    % Calculate accuracy for classifier with shuffled labels
    permuted_accuracy(k) = mean(1-accuracy_shuffled);
end

% P value is the fraction of times that the classifier behaved better in random enviorment 
% (aka fraction of time that a classifier with shuffled labels performed better than the original classifier)
p_value = (sum(permuted_accuracy >= actual_accuracy) + 1) / (num_permutations + 1);
disp(['SVM permutation test 2 p-value :', num2str(p_value,2)])

if p_value < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier has found a true connection in the data.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
% Binomial test
true_labels = balanced_eeg_data_test(:,end);
predicted_labels = predicted;

% Find number of successes k and number of independent trials n
num_trials = numel(true_labels);  % Number of trials n
num_success = sum(true_labels == predicted_labels);  % Number of successful predictions k

% Hypothesized probability (chance level, 0.5)
p_hypothesized = 0.5;

% Perform binomial test
p_value_binomial = binocdf(num_success - 1, num_trials, p_hypothesized, 'upper');

disp(['SVM binomial test p-value :', num2str(p_value_binomial,2)])

if p_value_binomial < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier performs above chance.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
fprintf("=======================================================================================\n")
%% Train classifier
%------------------------------------------------------------------------------------------------
% EMG

%------------------------------------------------------------------------------------------------
% Oversampling: https://se.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote
% 
% Permutation test: https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf
%------------------------------------------------------------------------------------------------

% Fix class imbalance with Synthetic Minority Over-sampling Technique (SMOTE)
[smote_data, smote_label, ~, ~] = smote(emg_features(:, 1:end-1),[], 5, 'Class', emg_features(:,end));
balanced_emg_data = [smote_data smote_label];

% 80 train 20 test
cv = cvpartition(size(balanced_emg_data,1),'HoldOut',0.2);
idx = cv.test;
balanced_emg_data_train = balanced_emg_data(~idx,:);
balanced_emg_data_test = balanced_emg_data(idx,:);

% Train classifier using 5 fold cross validation
% CHANGED TO 'pseudolinear' from 'linear' incase one class has zero variance
emg_classifier = fitcdiscr(balanced_emg_data_train(:,1:end-1), balanced_emg_data_train(:,end),"DiscrimType","pseudolinear","CrossVal","on","KFold",5); % LDA
save("trained_classifiers\emg_classifier.mat","emg_classifier");

fprintf("=======================================================================================\n")
% Accuracy
% Accuracy for each fold
formatSpec = '%.2f';
accuracy_emg_fold = kfoldLoss(emg_classifier, 'Mode', 'individual', 'LossFun', 'classiferror');
disp(['LDA 1st fold accuracy : ', num2str((1-accuracy_emg_fold(1))*100, formatSpec), '%']);
disp(['LDA 2nd fold accuracy : ', num2str((1-accuracy_emg_fold(2))*100, formatSpec), '%']);
disp(['LDA 3rd fold accuracy : ', num2str((1-accuracy_emg_fold(3))*100, formatSpec), '%']);
disp(['LDA 4th fold accuracy : ', num2str((1-accuracy_emg_fold(4))*100, formatSpec), '%']);
disp(['LDA 5th fold accuracy : ', num2str((1-accuracy_emg_fold(5))*100, formatSpec), '%']);

% Average accuracy over all folds
disp(['LDA average accuracy : ', num2str((mean(1-accuracy_emg_fold))*100, formatSpec), '% (+-', num2str(std(1-accuracy_emg_fold)*100, formatSpec), '%)']);

% Confusion matrix
% Predict on test set
predicted = predict(emg_classifier.Trained{1}, balanced_emg_data_test(:,1:end-1));
emg_confusion_matrix = confusionmat(balanced_emg_data_test(:,end), predicted);
figure
confusionchart(emg_confusion_matrix)

% Statistical tests
%------------------------------------------------------------------------------------------------
% Permutation test 1 (shuffle predicted labels)
true_labels = balanced_emg_data_test(:,end);
predicted_labels = predicted;

% Calculate accuracy for original classifier
actual_accuracy = sum(true_labels == predicted_labels,'all')/numel(predicted_labels);

num_permutations = 100;
permuted_accuracy = zeros(1,num_permutations);
for k = 1:num_permutations
    % Shuffle predicted labels
    shuffled_labels = predicted_labels(randperm(length(predicted_labels)));

    % Calculate accuracy when predicted labels have been shuffled
    permuted_accuracy(k) = sum(true_labels == shuffled_labels,'all')/numel(shuffled_labels);
end

% P value is the fraction of times that the shuffled predicted labels
% performed better than actual predicted labels 
p_value = (sum(permuted_accuracy >= actual_accuracy) + 1) / (num_permutations + 1);
disp(['LDA permutation test 1 p-value :', num2str(p_value,2)])

if p_value < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier performance is better than random chance.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
% Permutation test 2 (shuffle true labels)
true_labels = balanced_emg_data_train(:,end);
data_set = balanced_emg_data_train(:,1:end-1);

% Calculate accuracy for original classifier
actual_accuracy = mean(1-accuracy_emg_fold);

num_permutations = 100;
permuted_accuracy = zeros(1,num_permutations);
for k = 1:num_permutations
    % Shuffle the true labels and train classifier when labels no longer has a true connection to the data
    shuffled_true = true_labels(randperm(length(true_labels)));
    % CHANGED TO 'pseudolinear' from 'linear' because one class had zero variance
    perm_classifier = fitcdiscr(data_set, shuffled_true,"DiscrimType","pseudolinear","CrossVal","on","KFold",5); % LDA

    % CV accuracy
    accuracy_shuffled = kfoldLoss(perm_classifier, 'Mode', 'individual', 'LossFun', 'classiferror');

    % Calculate accuracy for classifier with shuffled labels
    permuted_accuracy(k) = mean(1-accuracy_shuffled);
end

% P value is the fraction of times that the classifier behaved better in random enviorment 
% (aka fraction of time that a classifier with shuffled labels performed better than the original classifier)
p_value = (sum(permuted_accuracy >= actual_accuracy) + 1) / (num_permutations + 1);
disp(['LDA permutation test 2 p-value :', num2str(p_value,2)])

if p_value < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier has found a true connection in the data.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
fprintf("=======================================================================================\n")