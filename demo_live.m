%% Hammering Test by Deep Learning
%
% This demo shows simple application of deep learning for signals.
% You can try the workflow which includes audio data acquisition,preprocessing,
% network creation,and visualization.
% Please run each section
%
% [Reference Document]
% https://jp.mathworks.com/help/deeplearning/examples/deep-learning-speech-recognition.html?lang=en
%
% [Required Products]
%  Signal Processing Toolbox
%  DSP System Toolbox       
%  Audio Toolbox
%  Deep Learning Toolbox    
%  (Optional)Parallel Computing Toolbox


%% Settings for peak detection
fs = 8820;              % Sampling frequency
segmentDuration = 0.25; % Length of signal[s]
pkh  = 0.05;            % Minimum peak height to be detected;

%% Acquire sound data of normal parts (Please hit the normal part repeatedlyÅj
% Close window to finish this process
savedir = 'livedata/normal';
samplePeaks(savedir,fs,segmentDuration,pkh);

%% Acquire sound data of abnormal parts (Please hit the normal part repeatedlyÅj
% Close window to finish this process
savedir = 'livedata/abnormal';
samplePeaks(savedir,fs,segmentDuration,pkh);

%% Acquire sound data of backgrounds (roughly 40 data is acquired after leaving for 10 secondsÅj
% Close window to finish this process
savedir = 'livedata/background';
sampleBG(savedir,fs,segmentDuration)

%% Create audio datastore
datasetdir = 'livedata';
ads = audioDatastore(datasetdir,'IncludeSubfolders',true,'LabelSource','foldernames');

% Check file count for each label
countEachLabel(ads)

%% Split Data into Training(70%), Validation(10%), and Test(20%) Sets
[adsTrain,adsValid,adsTest] = splitEachLabel(ads,0.7,0.1);

%% Compute Speech Spectrograms 
segmentDuration = 0.25; % Duration of each speech clip[s]
frameDuration = 0.01;   % Duration of each frame for spectrogram calculation[s]
hopDuration = 0.002;    % Time step between each column of the spectrogram
numBands = 40;          % The number of log-mel filters and equals the height of each spectrogram
epsil = 1e-6;

% Calculate spectrograms
STrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
STrain = log10(STrain + epsil);
SValid = speechSpectrograms(adsValid,segmentDuration,frameDuration,hopDuration,numBands);
SValid = log10(SValid + epsil);
STest  = speechSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
STest  = log10(STest + epsil);

% Set label data
YTrain = adsTrain.Labels;
YValid = adsValid.Labels;
YTest  = adsTest.Labels;

%% Visualize Data
reset(adsTrain);

idx  = [find(adsTrain.Labels == 'normal',1), ...
        find(adsTrain.Labels == 'abnormal',1) ...
        find(adsTrain.Labels == 'background',1)];

specMin = min(STrain(:));
specMax = max(STrain(:));
for i = 1:numel(idx)
    
    % Plot audio data
    subplot(2,3,i)
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    t  = (0:size(x)-1)/fs;
    plot(t,x); xlim([0,0.25]); ylim([-1 1]);
    grid on; xlabel('êUïù'); ylabel('Time[s]'); 
    title(adsTrain.Labels(idx(i)))
    
    % Show spectrograms
    subplot(2,3,i+3)
    pcolor(STrain(:,:,idx(i)))
    caxis([specMin+2 specMax])
    shading flat
    
    % Play sample data
    sound(x)
    
    pause(1)
end


%% Add Data Augmentation
sz = size(STrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,STrain,YTrain, ...
    'DataAugmentation',augmenter);

%% Define Neural Network Architecture
classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1 13])
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% Training Options
miniBatchSize = 64;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{SValid,YValid}, ...
    'ValidationFrequency',validationFrequency+1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.01, ...
    'LearnRateDropPeriod',20);

%% Train Network
trainedNet = trainNetwork(augimdsTrain,layers,options);

%% Evaluate Trained Network
YPred= classify(trainedNet,STest);
figure('Units','normalized');
cm = confusionchart(YTest,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%% Detect Commands Using Streaming Audio from Microphone

% If you want use your Web camera to show , connect the camera to your PC and set 'useCamera' to true. 
useCamera = false;


% Paremeter Settings
%fs = 8820; % Sampling frequency
%segmentDuration = 0.25; % Duration of each speech clip[s]
samplesPerFrame = ceil(fs*segmentDuration);
frameDuration = 0.01;  
hopDuration = 0.002;   
numBands = 40;        
epsil = 1e-6;

% Create audio reader object
audioIn = audioDeviceReader(fs, samplesPerFrame);

% Connect to web camera
if useCamera
    cam = webcam(2);
    hcam = figure('Units','normalized','Position',[0.5 0.0 0.5 1]);
    axcam = axes(hcam);
end

% Visualize results
h = figure('Units','normalized','Position',[0.0 0.0 0.5 1]);
specMin = -6;     % min(STrain(:));
specMax = 0.6184; % max(STrain(:));
while ishandle(h)
    
    x = audioIn();
    
    % Calculate spectrograms
    frameLength = round(frameDuration*fs);
    hopLength = round(hopDuration*fs);
    numHops = ceil((segmentDuration - frameDuration)/hopDuration);
    
    spec = melSpectrogram(x,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength - hopLength, ...
        'FFTLength',1024, ...
        'NumBands',numBands);
        %'FrequencyRange',[50,7000]);
    epsil = 1e-6;
        
    % Padding
    X = zeros([numBands,numHops],'single');
    w = size(spec,2);
    left = floor((numHops-w)/2)+1;
    ind = left:left+w-1;
    X(:,ind) = spec;
    X = log10(X + epsil);
    
    % Classify the current spectrogram
    [YPredicted,probs] = classify(trainedNet,X);
    
    % Plot the current waveform and spectrogram.
    
    subplot(2,1,1);
    t = (0:size(x,1)-1)/fs;   
    plot(t,x)
    ylim([-1 1]); grid on; xlabel('Time[s]');
    
    subplot(2,1,2)
    pcolor(X)
    caxis([specMin+2 specMax])
    shading flat
    
    % Show the predicted label as title of the figure 
    subplot(2,1,1);
    if YPredicted == "background" 
       title(" ")
    else
       title(string(YPredicted),'FontSize',20)
    end
    
    if useCamera
        img = snapshot(cam);
        imshow(img,'Parent',axcam)
    end
    
    drawnow
    
end

% Release objects
release(audioIn);
if useCamera
    clear cam
end


%% Copyright 2019 The MathWorks, Inc.