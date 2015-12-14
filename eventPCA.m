function missVoxel = eventPCA(numPC, train,testProv,missIdx,provideIdx)

numTest = size(testProv,1);
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    % learn weight
    cprov = train(:,provideIdx)';
    w = cprov\testProv(i,:)';
    
    % predict missing value
    cmiss = train(:,missIdx)';
    missVoxel(i,:) = (cmiss*w)';
end

end