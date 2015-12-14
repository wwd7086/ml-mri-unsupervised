clear;
close all;

% load datas
load('full.mat');
load('provideIdx.mat');
load('missIdx.mat');
load('provideData_1000.mat');

% load libs
addpath spams-matlab;
start_spams;
addpath KMeans++;

%% inputs
isTest = true;
if isTest
    doTest = true;
    doGen = false;
    train = X2train;
    testProv = X2test(:,provideIdx);
    testMiss = X2test(:,missIdx);
else
    doTest = false;
    doGen = true;
    train = [X2train;X2test];
    testProv = provideData;
end

%% learn PCA from full voxel images
%missVoxel = simplePCA(80,train,testProv,missIdx,provideIdx);
%missVoxel = kmeanRecon(200,train,testProv,missIdx,provideIdx);
missVoxel = superSparse(200,train,testProv,missIdx,provideIdx);

%% calculate erro
if doTest
    errs = missVoxel - testMiss;
    rmse = sqrt(mean(errs(:).^2))
end

%% generate csv
if doGen
    csvwrite('prediction.csv', missVoxel);
end
