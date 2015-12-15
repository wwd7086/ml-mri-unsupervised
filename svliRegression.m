function missVoxel = svliRegression(numPC, train,testProv,missIdx,provideIdx)
% train
trainProv = train(:,provideIdx);
trainMiss = train(:,missIdx);
model = svmtrain(trainMiss,trainProv,'-s 3 -t 0 -c 100 -p 10 -h 0');
%trueVoxel = zeros(size(testProv,1),size(missIdx,2));
%missVoxel = svmpredict(trueVoxel,testProv,model);
%save('svrmodel.mat','model');
%load svrmodel.mat;

% predict
sMiss = trainMiss(model.sv_indices,:);
sProv = trainProv(model.sv_indices,:);

numTest = size(testProv,1);
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    k = sProv * testProv(i,:)';
    k = k.*model.sv_coef;
    %k = k./sum(k);
    v = sum(bsxfun(@times,k,sMiss),1);
    missVoxel(i,:) = v;
end

end
