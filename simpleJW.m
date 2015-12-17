function missVoxel = simpleJW(numPC, train,testProv,missIdx,provideIdx)

%% SVR on each feature

% % center data
% trainMean = mean(train,1);
% train = bsxfun(@minus, train, trainMean);
% 
% [~,~,v] = svd(train);
% pc = v(:,1:numPC);

% normalize data
norm_train = train - min(train(:));
norm_train = norm_train ./ max(norm_train(:));
% train_norm = norm(train);
% norm_train = train / train_norm;
% norm_train = train - min(train(:));
% norm_train = normr(norm_train);

% norm_testProv = testProv - min(testProv(:));
% norm_testProv = norm_testProv ./ max(norm_testProv(:));
% testProv_norm = norm(testProv);
% norm_testProv = testProv / testProv_norm;

% Number of features to train (2731)
numTest = size(missIdx,2);
% Number of data points / observations (1000)
numData = size(testProv,1);
% Fill out mixing voxels (1000 x 2731)
missVoxel = zeros(numData,numTest);

% Get provided section and normalize it
train_provided = train(:,provideIdx);
train_provided = train_provided - min(train_provided(:));
train_provided = train_provided ./ max(train_provided(:));

% Don't normalize missing section
train_missing  = train(:,missIdx);

% Get provided DATA section and normalize it
test_provided = testProv - min(testProv(:));
test_provided = test_provided ./ max(test_provided(:));

tic
%parfor i=1:5
parfor i=1:numTest
    fprintf('==== Loop: %d ====\n', i);
    %% Support vector regression
    % Get block of given data from training
    %A_prov = norm_train(:,provideIdx);
    % block of given 'missing' data from test is testProv
    
    %size(A_prov)   % 1502 x 3172
    %size(testProv) % 1000 x 3172
    %size(norm_train(:,missIdx))  % 1502 2731   
    
    size(train_provided(:,i))
    size(train_missing(:,i))
    size(test_provided)
    
    % 1502 x 3172   x     1000 x 1     
    
    %model = svmtrain(testProv(i,:)', A_prov', '-s 3 -t 2 -c 97 -p 0.2');
    model = svmtrain(train_missing(:,i), train_provided, '-s 3 -t 2 -c 97 -p 0.2');
    %model = svmtrain(train_missing(:,i), train_provided(:,i), '-s 3 -t 2 -c 97 -p 0.2');
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 80 score 68.0
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 65 score 67.8
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.01 -h 0' pca 65
    % '-s 3 -t 2 -g 0.05 -c 300 -p 0.1 -h 0' score 77.8

    %size(zeros(1,size(testProv,1)))
    %size(test_provided(:,i))
    
    [predicted,~,~] = svmpredict(zeros(size(testProv,1),1), test_provided, model);
    %[predicted,~,~] = svmpredict(zeros(size(testProv,1),1), test_provided(:,i), model);
    
    missVoxel(:,i) = predicted;
end
toc

end