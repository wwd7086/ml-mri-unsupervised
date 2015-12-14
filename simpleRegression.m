function missVoxel = simpleRegression(numPC, train,testProv,missIdx,provideIdx)

trainProv = train(:,provideIdx);
trainMiss = train(:,missIdx);
%w = trainProv\trainMiss;
w = mvregress(trainProv,trainMiss);
missVoxel = testProv * w;

end
