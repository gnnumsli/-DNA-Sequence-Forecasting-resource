%% 通用matlab脚本三连
clear
clc
close all
 
%% 加载序列数据
% 加载日语元音训练数据。XTrain 是包含 270 个不同长度的 12 维序列的元胞数组。
% Y 是对应于九个说话者的标签 "1"、"2"、...、"9" 的分类向量。
% XTrain 中的条目是具有 12 行（每个特征一行）和不同列数（每个时间步一列）的矩阵。
[XTrain,YTrain] = japaneseVowelsTrainData;
XTrain(1:5)
 
%% 在绘图中可视化第一个时序。每行对应一个特征。
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
legend("Feature " + string(1:12),'Location','northeastoutside')
%% 准备要填充的数据
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
%% 按序列长度对数据进行排序。
[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);
%% 在条形图中查看排序的序列长度。
figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
%% 选择小批量大小 27 以均匀划分训练数据，并减少小批量中的填充量。下图说明了添加到序列中的填充。
miniBatchSize = 27;
%% 定义 LSTM 网络架构
inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
maxEpochs = 100;
miniBatchSize = 27;
%% 指定训练选项
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%% 训练 LSTM 网络
net = trainNetwork(XTrain,YTrain,layers,options);
%% 测试 LSTM 网络
[XTest,YTest] = japaneseVowelsTestData;
XTest(1:3)
%% LSTM 网络 net 已使用相似长度的小批量序列进行训练
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);
%% 对测试数据进行分类
miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% 计算预测值的分类准确度。
acc = sum(YPred == YTest)./numel(YTest)