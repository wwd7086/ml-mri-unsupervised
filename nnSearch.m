function missVoxel = nnSearch(numPC, train,testProv,missIdx,provideIdx)

    numTest = size(testProv,1);
    trainProv = train(:,provideIdx);
    missVoxel = zeros(numTest,size(missIdx,2));

    for i=1:numTest
        % find NN
        diff = bsxfun(@minus, trainProv, testProv(i,:));
        diff = diff.^2;
        diff = mean(diff,2);
        [diffSort,ind] = sort(diff);
        
        % reconstruct
        missVoxel(i,:) = train(ind(1),missIdx);
    end
    
end
