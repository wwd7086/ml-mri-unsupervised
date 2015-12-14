function missVoxel = kmeanRecon(numPC, train,testProv,missIdx,provideIdx)

% learn the dictionary
nReplicates     = 10;
[~,centers,~,~] = kmeans_sparsified( train', numPC,'ColumnSamples',true,...
    'Display','off','Replicates',nReplicates,'Sparsify',false, 'start','++');

%centers k*p
numTest = size(testProv,1);
missVoxel = zeros(numTest,size(missIdx,2));
for i=1:numTest
    % learn weight
    cprov = centers(:,provideIdx)';
    w = cprov\(testProv(i,:))';
    
    % predict missing value
    cmiss = centers(:,missIdx)';
    missVoxel(i,:) = (cmiss*w)';
end

end
