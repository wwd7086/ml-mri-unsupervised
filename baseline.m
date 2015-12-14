clear;
close all;

load('full.mat');
load('provideIdx.mat');
load('missIdx.mat');
load('provideData_1000.mat');

%% inputs
doTest = true;
doGen = false;

train = X2train;
testProv = X2test(:,provideIdx);
testMiss = X2test(:,missIdx);

%train = [X2train;X2test];
%testProv = provideData;

%% learn PCA from full voxel images
%missVoxel = simplePCA(80,train,testProv,missIdx,provideIdx);
missVoxel = kmeanRecon(200,train,testProv,missIdx,provideIdx);

%% calculate erro
if doTest
    errs = missVoxel - testMiss;
    rmse = sqrt(mean(errs(:).^2))
end

%% generate csv
if doGen
    csvwrite('prediction.csv', missVoxel);
end