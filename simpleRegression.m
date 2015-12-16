function missVoxel = simpleRegression(numPC, train,testProv,missIdx,provideIdx)

trainProv = train(:,provideIdx);
trainMiss = train(:,missIdx);
w = trainProv\trainMiss;

end
