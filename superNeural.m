function missVoxel = superNeural(numPC, vtrain,testProv,missIdx,provideIdx)

trainProv = vtrain(:,provideIdx)';
trainMiss = vtrain(:,missIdx)';

net = feedforwardnet(10);
net = train(net,trainProv,trainMiss);
missVoxel = net(testProv')';

end
