function missVoxel = svRegression(numPC, train,testProv,missIdx,provideIdx)
% train
trainProv = train(:,provideIdx);
trainMiss = train(:,missIdx);
model = svmtrain(trainMiss,trainProv,'-s 3 -t 2 -g 10 -c 100 -p 0.1 -h 0');

trueVoxel = zeros(size(testProv,1),1);
missVoxel = svmpredict(trueVoxel,testProv,model);
%save('svrmodel.mat','model');
%load svrmodel.mat;

% predict
sMiss = trainMiss(model.sv_indices,:);
sProv = trainProv(model.sv_indices,:);

gamma = model.Parameters(4);
numTest = size(testProv,1);
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    diff = bsxfun(@minus,sProv,testProv(i,:));
    k = exp(-sum(diff.^2,2)*gamma);
    k = k.*model.sv_coef;
    %k = k./sum(k);
    v = sum(bsxfun(@times,k,sMiss),1);
    missVoxel(i,:) = v;
end

end
