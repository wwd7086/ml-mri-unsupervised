function missVoxel = simpleSVR(numPC, train,testProv,missIdx,provideIdx)

%% PCA with SVR

%% learn PCA from full voxel images
numTest = size(testProv,1);

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

% Fill out mixing voxels
missVoxel = zeros(numTest,size(missIdx,2));

%size(norm_train)

tic
%parfor i=1:5
parfor i=1:numTest
    fprintf('==== Loop: %d ====\n', i);
    %% Support vector regression
    % Get block of given data from training
    A_prov = norm_train(:,provideIdx);
    % block of given 'missing' data from test is testProv
    
    size(A_prov)
    size(testProv)
    size(norm_train(:,missIdx))    
    
%     model = svmtrain(norm_testProv(i,:)', A_prov', '-s 3 -t 2 -g 0.01 -c 300 -p 0.1 -h 0');
    model = svmtrain(testProv(i,:)', A_prov', '-s 3 -t 2 -c 99 -p 0.2');
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 80 score 68.0
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 65 score 67.8
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.01 -h 0' pca 65
    % '-s 3 -t 2 -g 0.05 -c 300 -p 0.1 -h 0' score 77.8

    [predicted,~,~] = svmpredict(zeros(size(missIdx,2),1), norm_train(:,missIdx)', model);
    
    missVoxel(i,:) = predicted;
end
toc

end