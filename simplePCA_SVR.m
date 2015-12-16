function missVoxel = simplePCA_SVR(numPC, train,testProv,missIdx,provideIdx)

%% PCA with SVR

%% learn PCA from full voxel images
numTest = size(testProv,1);

% center data
trainMean = mean(train,1);
train = bsxfun(@minus, train, trainMean);

[~,~,v] = svd(train);
pc = v(:,1:numPC);

% Fill out mixing voxels
missVoxel = zeros(numTest,size(missIdx,2));

for i=1:numTest
    %% Support vector regression
    % Get block of given data from training
    A_prov = pc(provideIdx,:);    
    % block of given 'missing' data from test is testProv
    model = svmtrain(testProv(i,:)', A_prov, '-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0');
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 80 score 68.0
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.1 -h 0' pca 65 score 67.8
    %'-s 3 -t 2 -g 0.05 -c 150 -p 0.01 -h 0' pca 65

    [predicted,~,~] = svmpredict(zeros(size(missIdx,2),1), pc(missIdx,:), model);
    
    missVoxel(i,:) = predicted;
end

end