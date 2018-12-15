function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

    numImages = size(convolvedFeatures, 4);
    numFilters = size(convolvedFeatures, 3);
    convolvedDim = size(convolvedFeatures, 1);
    poolRow = convolvedDim / poolDim;
    poolCol = convolvedDim / poolDim;
    pooledFeatures = zeros(poolRow, poolCol, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
    W = 1/(poolDim^2) * ones(poolDim);
    
    for ImagesNo = 1:numImages
        for FilterNo = 1: numFilters
            for col = 1:poolCol
                for row = 1:poolRow
                    poolingImage = squeeze(convolvedFeatures((row-1)*poolDim+1:row*poolDim,...
                        (col-1)*poolDim+1:col*poolDim, FilterNo, ImagesNo));
                    pooledFeatures(row,col,FilterNo, ImagesNo) = conv2(poolingImage, W, 'valid');
                end
            end
        end
    end
            
end

