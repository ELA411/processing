%% Code for training EEG and EMG classifiers
%========================================================================================================
% Author: Carl Larsson
% Description: Performs all processing of EEG and EMG data including
% training of classifier and evaluation of trained model
% Use: Alter the search paths and files to the desired paths and files, change eeg_fs and
% emg_fs in the "Load data" section to the sample rate of the EEG and EMG
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
raw_eeg_data = load("data_sets\EEG_10.txt"); % EEG data set (expected column format: channels, labels, package ID, timestamp. each row is expected to be subsequent observations)
eeg_fs=200; % EEG sample rate
raw_emg_data = load("data_sets\EMG_10.txt"); % EMG data set (expected column format: channels, labels, package ID, timestamp. each row is expected to be subsequent observations)
emg_fs=1000; % EMG sample rate
%% Display data lost
%------------------------------------------------------------------------------------------------
% EEG
fprintf("=======================================================================================\n")
data_lost = 0;
last_last_id = raw_eeg_data(1,6);
last_id = raw_eeg_data(2,6);
for i = 3:length(raw_eeg_data)
    % EEG samples are sent two at a time with the same package ID (diplicates)
    if (raw_eeg_data(i,6) ~= last_id) && (last_id ~= last_last_id) % Duplicate half lost
        data_lost = data_lost + 1;
    elseif raw_eeg_data(i,6) == last_id % Duplicate
    % EEG package ID goes from 100 to 200
    elseif raw_eeg_data(i,6) == 100 + mod((last_id + 1) - 100, 100) % Next package ID is previous + 1
    else
        data_lost = data_lost + 2; % Package ID skipped, a duplicate pair is lost
    end
    last_last_id = last_id;
    last_id = raw_eeg_data(i,6);
end
disp(['EEG Data lost: ', num2str(data_lost)]);
%------------------------------------------------------------------------------------------------
% EMG
data_lost = 0;
for i = 2:length(raw_emg_data)
    % If next package doesn't have last package ID + 1, then a package has been lost
    % EMG package ID is mod 1000
    if raw_emg_data(i,4) ~= mod(raw_emg_data(i-1,4) + 1, 1000)
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
%------------------------------------------------------------------------------------------------
% EMG
fprintf("EMG data quality")
emg_quality = extract(sFE,raw_emg_data(:, 1:2));
fprintf('\nChannel 1:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',emg_quality(:,1,1),emg_quality(:,2,1), emg_quality(:,3,1));
fprintf('\nChannel 2:\nSNR:\t %f\nSINAD:\t %f\nTHD:\t %f\n',emg_quality(:,1,2),emg_quality(:,2,2), emg_quality(:,3,2));
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

%{
% Median filtering of the baseline wandering which usually is of a frequency of about 0.5Hz, but can be higher with movement.
% Baseline wandering is usually caused by respiration, electrodes(like poor contact or internal changes) and motion.
% The values 24 and 80 is for a sampling frequency of 500Hz.
s_filt_1=medfilt1(raw_eeg_data(:,1:4),(eeg_fs/500)*24);
s_filt_2=medfilt1(s_filt_1,(eeg_fs/500)*80);
filtered_eeg_data = raw_eeg_data(:,1:4) - s_filt_2;
%}

% Remove baseline wandering and DC offset
% 4th order Butterworth highpass filter 0.1hz cut off frequency.
[n,d] = butter(4,[0.1 99]/(eeg_fs/2),"bandpass");
filtered_eeg_data = filter(n,d,raw_eeg_data(:,1:4));

% Removal of 50Hz noise and all of it's harmonics up to 100Hz. 
% The noise is caused by magnetic fields generated by powerlines.
% Using a IIR notch filter of the fourth order as it's at that point the 50Hz noise peak on most signals practicaly becomes zero if a quality factor of about 25-35 is being used.
fo = 4;     % Filter order.
cf = 50/(eeg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
for k = 1:2
    notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * k,qf,pbr);
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
% CSP filter data
%filtered_eeg_data = transpose(W'*transpose(filtered_eeg_data));


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
% 20–500Hz fourth-order Butterworth bandpass filter.
[n,d] = butter(4,[20 499]/(emg_fs/2),"bandpass");
filtered_emg_data = filter(n,d,raw_emg_data(:,1:2));

% Removal of 50Hz noise and all of it's harmonics up to 150Hz. 
% The noise is caused by magnetic fields generated by powerlines.
% Using a IIR notch filter of the fourth order as it's at that point the 50Hz noise peak on most signals practicaly becomes zero if a quality factor of about 25-35 is being used.
fo = 4;     % Filter order.
cf = 50/(emg_fs/2); % Center frequency, value has to be between 0 and 1, where 1 is pi which is the Nyquist frequency which for our signal is Fs/2 = 500Hz.
qf = 30;   % Quality factor.
pbr = 1;   % Passband ripple, dB.
for k = 1:3
    notchSpecs  = fdesign.notch('N,F0,Q,Ap',fo,cf * k,qf,pbr);
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
[eeg_1, ~] = buffer(filtered_eeg_data(:,1),window_size*eeg_fs, overlap*eeg_fs, 'nodelay'); % Channel 1
[eeg_2, ~] = buffer(filtered_eeg_data(:,2),window_size*eeg_fs, overlap*eeg_fs, 'nodelay'); % Channel 2
[eeg_3, ~] = buffer(filtered_eeg_data(:,3),window_size*eeg_fs, overlap*eeg_fs, 'nodelay'); % Channel 3
[eeg_4, ~] = buffer(filtered_eeg_data(:,4),window_size*eeg_fs, overlap*eeg_fs, 'nodelay'); % Channel 4
[eeg_label, ~] = buffer(filtered_eeg_data(:,5),window_size*eeg_fs, overlap*eeg_fs, 'nodelay'); % Labels
%------------------------------------------------------------------------------------------------
% EMG
% Window EMG signal into 250ms windows with 50ms overlap
window_size = 0.250;                        % window size s
overlap = 0.050;                            % window overlap s
[emg_1, ~] = buffer(filtered_emg_data(:,1),window_size*emg_fs, overlap*emg_fs, "nodelay"); % Channel 1
[emg_2, ~] = buffer(filtered_emg_data(:,2),window_size*emg_fs, overlap*emg_fs, "nodelay"); % Channel 2
[emg_label, ~] = buffer(filtered_emg_data(:,3),window_size*emg_fs, overlap*emg_fs, "nodelay"); % Labels
%% Feature extraction
%------------------------------------------------------------------------------------------------
% EEG

% Extract features from each window and channel
[~, col_size] = size(eeg_1);
eeg_1_features = zeros(col_size, 4); % Create matrix containing all extracted features from each window beforehand
for i=1:col_size
    eeg_channels_window = [eeg_1(:,i), eeg_2(:,i), eeg_3(:,i), eeg_4(:,i)]; % Window with all channels
    csp_eeg = transpose(W'*transpose(eeg_channels_window)); % CSP filter window
    log_pow = log(bandpower(csp_eeg,eeg_fs,[0 eeg_fs/2])); % Log power

    eeg_1_features(i,:) = [log_pow];
end

% Labels
% Labels for each window is calculated as the majority class of that window
[~, col_size] = size(eeg_label);
eeg_label_window = zeros(col_size, 1);
for i=1:col_size
    eeg_label_window(i,:) = round(mean(eeg_label(:,i)));
end

% 1 channel 1 features
% 2 channel 2 features
% 3 channel 3 features
% 4 channel 4 features
% 5 labels
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
for i=1:col_size
    f_mav = jfemg('mav', emg_1(:,i)); % Mean absolut value
    f_wl = jfemg('wl', emg_1(:,i)); % Waveform length
    f_zc = jfemg('zc', emg_1(:,i)); % Zero crossing
    f_ssc = jfemg('ssc', emg_1(:,i)); % Slope sign change
    opts.order = 1; % Defines output dimension
    f_ar = jfemg('ar', emg_1(:,i), opts); % Auto regressive

    emg_1_features(i,:) = [f_mav, f_wl, f_zc, f_ssc, f_ar];
end

[~, col_size] = size(emg_2);
emg_2_features = zeros(col_size, 5); % Create matrix containing all extracted features from each window beforehand
for i=1:col_size
    f_mav = jfemg('mav', emg_2(:,i)); % Mean absolut value
    f_wl = jfemg('wl', emg_2(:,i)); % Waveform length
    f_zc = jfemg('zc', emg_2(:,i)); % Zero crossing
    f_ssc = jfemg('ssc', emg_2(:,i)); % Slope sign change
    opts.order = 1; % Defines output dimension
    f_ar = jfemg('ar', emg_2(:,i), opts); % Auto regressive

    emg_2_features(i,:) = [f_mav, f_wl, f_zc, f_ssc, f_ar];
end

% Labels
% Labels for each window is calculated as the majority class of that window
[~, col_size] = size(emg_label);
emg_label_window = zeros(col_size, 1);
for i=1:col_size
    emg_label_window(i,:) = round(mean(emg_label(:,i)));
end

% 1:5 channel 1 features
% 6:10 channel 2 features
% 11 labels
emg_features = [emg_1_features emg_2_features emg_label_window];
%% Train classifier
%------------------------------------------------------------------------------------------------
% Oversampling: https://se.mathworks.com/matlabcentral/fileexchange/75401-synthetic-minority-over-sampling-technique-smote
% 
% Save and load models: https://se.mathworks.com/matlabcentral/answers/264160-how-to-save-and-reuse-a-trained-neural-network
% 
% Permutation test: https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf
% 
% Binomial test: https://www.sciencedirect.com/science/article/pii/S2213158214000485
%------------------------------------------------------------------------------------------------

%------------------------------------------------------------------------------------------------
% EEG
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

num_permutations = 10000;
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

num_permutations = 10000;
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
num_trials = numel(true_labels);  % Number of trials
num_success = sum(true_labels == predicted_labels);  % Number of successful predictions

% Hypothesized probability (e.g., chance level, 0.5 for a binary classifier)
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
% Fix class imbalance with Synthetic Minority Over-sampling Technique (SMOTE)
[smote_data, smote_label, ~, ~] = smote(emg_features(:, 1:end-1),[], 5, 'Class', emg_features(:,end));
balanced_emg_data = [smote_data smote_label];

% 80 train 20 test
cv = cvpartition(size(balanced_emg_data,1),'HoldOut',0.2);
idx = cv.test;
balanced_emg_data_train = balanced_emg_data(~idx,:);
balanced_emg_data_test = balanced_emg_data(idx,:);

% Train classifier using 5 fold cross validation
% CHANGED TO 'pseudolinear' from 'linear' because one class had zero variance
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

num_permutations = 10000;
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
disp(['LDA permutation test p-value 1 :', num2str(p_value,2)])

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

num_permutations = 10000;
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
disp(['LDA permutation test p-value 2 :', num2str(p_value,2)])

if p_value < 0.05
    disp('Classifier performance is statistically significant. Null hypothesis is rejected, the classifier has found a true connection in the data.');
else
    disp('Classifier performance is not statistically significant. Null hypothesis is not rejected.');
end
%------------------------------------------------------------------------------------------------
fprintf("=======================================================================================\n")