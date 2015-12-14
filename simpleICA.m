function [ missVoxel ] = simpleICA(numPC, train, testProv, missIdx, provideIdx)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

addpath('../FastICA_25');

numTest = size(testProv,1);

icasig = fastica(train, 'numOfIC', numPC);
save('ica.mat', 'icasig');
icasig = icasig';

%% using PCA
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    % learn weight
    cprov = icasig(provideIdx,:);
    w = cprov\testProv(i,:)';
    
    % predict missing value
    cmiss = icasig(missIdx,:);
    missVoxel(i,:) = (cmiss*w)';
end

end

