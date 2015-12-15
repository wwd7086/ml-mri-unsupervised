function missVoxel = simpleRegression(numPC, train,testProv,missIdx,provideIdx)

trainProv = train(:,provideIdx);
trainProv = [ones(size(trainProv,1),1),trainProv];
trainMiss = train(:,missIdx);
w = trainProv\trainMiss;
%w = mvregress(trainProv,trainMiss,'algorithm','mvn');
missVoxel = [ones(size(testProv,1),1),testProv] * w;

end
