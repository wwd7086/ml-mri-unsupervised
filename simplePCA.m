function missVoxel = simplePCA(numPC, train,testProv,missIdx,provideIdx)

%% learn PCA from full voxel images
numTest = size(testProv,1);
[~,~,v] = svd(train);
pc = v(:,1:numPC);

%% using PCA
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    % learn weight
    cprov = pc(provideIdx,:);
    w = cprov\testProv(i,:)';
    
    % predict missing value
    cmiss = pc(missIdx,:);
    missVoxel(i,:) = (cmiss*w)';
end

end