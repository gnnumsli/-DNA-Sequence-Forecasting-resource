%% ͨ��matlab�ű�����
clear
clc
close all
 
%% ������������
% ��������Ԫ��ѵ�����ݡ�XTrain �ǰ��� 270 ����ͬ���ȵ� 12 ά���е�Ԫ�����顣
% Y �Ƕ�Ӧ�ھŸ�˵���ߵı�ǩ "1"��"2"��...��"9" �ķ���������
% XTrain �е���Ŀ�Ǿ��� 12 �У�ÿ������һ�У��Ͳ�ͬ������ÿ��ʱ�䲽һ�У��ľ���
[XTrain,YTrain] = japaneseVowelsTrainData;
XTrain(1:5)
 
%% �ڻ�ͼ�п��ӻ���һ��ʱ��ÿ�ж�Ӧһ��������
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
legend("Feature " + string(1:12),'Location','northeastoutside')
%% ׼��Ҫ��������
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
%% �����г��ȶ����ݽ�������
[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);
%% ������ͼ�в鿴��������г��ȡ�
figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
%% ѡ��С������С 27 �Ծ��Ȼ���ѵ�����ݣ�������С�����е����������ͼ˵������ӵ������е���䡣
miniBatchSize = 27;
%% ���� LSTM ����ܹ�
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
%% ָ��ѵ��ѡ��
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%% ѵ�� LSTM ����
net = trainNetwork(XTrain,YTrain,layers,options);
%% ���� LSTM ����
[XTest,YTest] = japaneseVowelsTestData;
XTest(1:3)
%% LSTM ���� net ��ʹ�����Ƴ��ȵ�С�������н���ѵ��
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);
%% �Բ������ݽ��з���
miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% ����Ԥ��ֵ�ķ���׼ȷ�ȡ�
acc = sum(YPred == YTest)./numel(YTest)