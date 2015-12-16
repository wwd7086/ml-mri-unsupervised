function missVoxel = simplePCA(numPC, train,testProv,missIdx,provideIdx)

%% learn PCA from full voxel images
numTest = size(testProv,1);

% center data
trainMean = mean(train,1);
train = bsxfun(@minus, train, trainMean);

[~,~,v] = svd(train);
pc = v(:,1:numPC);

%% using PCA
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    % learn weight
    cprov = pc(provideIdx,:);
    mprov = trainMean(1,provideIdx);
    w = cprov\(testProv(i,:)-mprov)';
    
    % predict missing value
    cmiss = pc(missIdx,:);
    mmiss = trainMean(1,missIdx);
    missVoxel(i,:) = (cmiss*w)'+mmiss;
end

end
