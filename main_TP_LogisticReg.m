%%   LOGISTIC REGRESSION CLASSIFIER
% ------------------------------------
%   LOGISTIC REGRESSION CLASSIFIER
% ------------------------------------

% Abdelhakim Benechehab

clear
clc
close all

%% STEP 1: DESCRIPTION

% Load the images and labels
load Horizontal_edges;

%imshow(images_croped{1});
imshow(horiz_edges{1});


%%
% Number of images of cutting edges
num_edges = length(horiz_edges);

%%
% Number of features (in the case of ShapeFeat it is 10)
num_features = 10;

%%
% Initialiation of matrix of descriptors. It will have a size (m x n), where
% m is the number of training patterns (i.e. elements) and n is the number 
% of features (i.e. the length of the feature vector which characterizes 
% the cutting edge).
X = zeros(num_edges, num_features);

%%
% Describe the images of the horizontal edges by calling the fGetShapeFeat 
% function
for i=1:num_edges
    %disp(['Describing image number ' num2str(i)]);
    
    % Get the i-th cutting edge
    edge = logical(horiz_edges{i}); % DON'T REMOVE
    
    
    % Compute the descriptors of the cutting edge usign the fGetShapeFeat
    % function
    desc_edge_i = fGetShapeFeat(edge);
    
    % Store the feature vector into the matrix X.
    X(i,:) = desc_edge_i;
end



%%
% Create the vector of labels Y. Y(j) will store the label of the curring
% edge represented by the feature vector contained in the j-th row of the 
% matrix X.
% The problem will be binary: class 0 correspond to a low or medium wear
% level, whereas class 1 correspond to a high wear level.
Y = labels(:,2)'>=2;

save('tool_descriptors.mat', 'X', 'Y');

%% STEP 2: CLASSIFICATION

%%
%PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE ANYTHING)

clear
clc
close all

load tool_descriptors;
%%
% X contains the training patterns (dimension 10)
% Y contains the class label of the patterns (i.e. Y(37) contains the label
% of the pattern X(37,:) ).

% Number of patterns (i.e., elements) and variables per pattern in this
% dataset
[num_patterns, num_features] = size(X);

%%
% Normalization of the data
mu_data = mean(X);
std_data = std(X);
X = (X-mu_data)./std_data;

%%
% Parameter that indicates the percentage of patterns that will be used for
% the training
p_train = 0.6;

%%
% SPLIT DATA INTO TRAINING AND TEST SETS

num_patterns_train = round(p_train*num_patterns);

indx_permutation = randperm(num_patterns);

indxs_train = indx_permutation(1:num_patterns_train);
indxs_test = indx_permutation(num_patterns_train+1:end);

X_train = X(indxs_train, :);
Y_train = Y(indxs_train);

X_test= X(indxs_test, :);
Y_test = Y(indxs_test);


%% PART 2.1: TRAINING OF THE CLASSIFIER AND CLASSIFICATION OF THE TEST SET

% Learning rate. Change it accordingly, depending on how the cost function
% evolve along the iterations
alpha = 0.1;

%%
% The function fTrain_LogisticReg implements the logistic regression 
% classifier. Open it and complete the code.

%%
% TRAINING
theta = fTrain_LogisticReg(X_train, Y_train, alpha);

%% 
% The first question we need to answer based on this figure is about the
% sufficiency of the number of iterations of the optimization algorithm we
% have fixed at 100. The answer is yes clearly, because if we look at the
% figure with alpha=2 -the most accurate- we can see that the cost function
% has almost stagnated so we have reached a certain stability.

%% 
% Considering the two other values of alpha, for alpha=6, we can say that we
% have a convergence problem as shown in the course material, thus the
% swings we observe.

%% 
% For alpha=0.1 the convergence is much slower than alpha=2 that's why we
% chose this last value as the optimal value of alpha.

%%
% CLASSIFICATION OF THE TEST SET
Y_test_hat = fClassify_LogisticReg(X_test, theta);

%%
% Assignation of the class
Y_test_asig = Y_test_hat>=0.5;

%% PART 2.2: PERFORMANCE OF THE CLASSIFIER: CALCULATION OF THE ACCURACY AND FSCORE

% Show confusion matrix
figure;
plotconfusion(Y_test, Y_test_asig);


%%
% ACCURACY AND F-SCORE
% ====================== YOUR CODE HERE ======================
% Cf = confusionmat(Y_test, Y_test_asig);
% 
% accuracy = trace(Cf)/sum(sum(Cf));
% 
% Precision = Cf(1,1)/(Cf(1,1)+Cf(2,1));
% 
% Recall = Cf(1,1)/(Cf(1,1)+Cf(1,2));
% 
% FScore = 2*((Precision*Recall)/(Precision+Recall));

Cfman = zeros(2,2);

for i = 1:81
   if (Y_test(i) == 1) && (Y_test_asig(i) == 1) 
       Cfman(1,1) = Cfman(1,1)+1;
   elseif (Y_test(i) == 0) && (Y_test_asig(i) == 1) 
       Cfman(2,1) = Cfman(2,1)+1;
   elseif (Y_test(i) == 1) && (Y_test_asig(i) == 0) 
       Cfman(1,2) = Cfman(1,2)+1;
   elseif (Y_test(i) == 0) && (Y_test_asig(i) == 0)
       Cfman(2,2) = Cfman(2,2)+1;
   end
end

Cf = Cfman;

accuracy = trace(Cf)/sum(sum(Cf));

Precision = Cf(1,1)/(Cf(1,1)+Cf(2,1));

Recall = Cf(1,1)/(Cf(1,1)+Cf(1,2));

FScore = 2*((Precision*Recall)/(Precision+Recall));

fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);

%% 
% The confusion matrix has strong values in its diagonal hence we can
% already tell that our classifier is not bad.

%% 
% Looking at the accuracy value (90%), We can assume that our classifier is 
% very efficient. But like we saw in the course material the accuracy rate 
% can be a very falsy criterion if our data is strongly asymmetric. 

%% 
%To get rid of any doubt we will check the value of the F-Score that takes
% in consideration the asymmetry of the problem, this indicator (0.6)
% confirms our pre-thoughts about our classifier, since the value is
% relatively high and that our data are quite unbalanced.




%% Model evaluation: Performance metrics (ROC analysis)

[TPR,FPR] = ROC(Y_test,Y_test_hat);

figure;
plot(FPR,TPR,[0,1],[0,1],'--')
title('Model evaluation: Performance metrics (ROC analysis)');
xlabel('FPR(1-specificity)');
ylabel('TPR(sensitivity)');
legend('ROC Classifier','Random classification');

%% 
% The curve we got is more than sufficient to tell that our classifier is
% robust, it is entirely above the first bicector. 

%% 
% To make sure we will calculate the value of its integral on the [0,1] 
% domain -the area under the curve-

[n, m] = size(TPR);
q = 0;
%Integral is computed using the rectangle rule
for i=1:(m-1)
    q=q+(FPR(i)-FPR(i+1))*TPR(i);
end

disp(q)

%% 
% The value obtained is very close to 1 thus our classifier is precize.


%% SVM Classifier

%HyperParameters optimization and kernelType selection
Modl = fitcsvm(X,Y,'KernelFunction','polynomial','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%tried on three kernels : linear, gaussian and polynomial

%%

Modlfinal = fitcsvm(X_train,Y_train,'KernelFunction','polynomial','kernelScale',25.872,'BoxConstraint',846.1);

[label,score] = predict(fitPosterior(compact(Modlfinal),X_train,Y_train),X_test);

Y_test_asig = label';

table(Y_test',Y_test_asig',score(:,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','PostProb'})

figure;
plotconfusion(Y_test, Y_test_asig);


Cfman = zeros(2,2);

for i = 1:81
   if (Y_test(i) == 1) && (Y_test_asig(i) == 1) 
       Cfman(1,1) = Cfman(1,1)+1;
   elseif (Y_test(i) == 0) && (Y_test_asig(i) == 1) 
       Cfman(2,1) = Cfman(2,1)+1;
   elseif (Y_test(i) == 1) && (Y_test_asig(i) == 0) 
       Cfman(1,2) = Cfman(1,2)+1;
   elseif (Y_test(i) == 0) && (Y_test_asig(i) == 0)
       Cfman(2,2) = Cfman(2,2)+1;
   end
end

Cf = Cfman;

accuracy = trace(Cf)/sum(sum(Cf));

Precision = Cf(1,1)/(Cf(1,1)+Cf(2,1));

Recall = Cf(1,1)/(Cf(1,1)+Cf(1,2));

FScore = 2*((Precision*Recall)/(Precision+Recall));

fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100);
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);

[TPR,FPR] = ROC(Y_test,score(:,2));

figure;
plot(FPR,TPR,[0,1],[0,1],'--')
title('Model evaluation: Performance metrics (ROC analysis)');
xlabel('FPR(1-specificity)');
ylabel('TPR(sensitivity)');
legend('ROC Classifier','Random classification');

%% Shallow Neural networks classifier

inputs = X';
targets = Y;

% Create a Pattern Recognition Network
hiddenLayerSize = [64 32 16 8];
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% View the Network
view(net)





