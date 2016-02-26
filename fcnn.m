function y1bag = fcnn(dataset)
addpath('fcnn')
k = strfind(dataset, '.mat');
if isempty(k)
    opts.dataDir = ['data/' dataset];
    opts.expDir = ['data/' dataset '/'];
    opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
else
    opts.imdbPath = dataset;
end
opts.train.batchSize = 12 ; % for dataset S123_label18
% opts.train.batchSize = 20 ;
opts.train.numEpochs =8;
% opts.train.numEpochs =32;
opts.train.continue = true ;
opts.train.useGpu = false ;
% opts.train.learningRate = 0.001 ;
opts.train.learningRate = [0.01*ones(1, 3) 0.001*ones(1, 25) 0.0001*ones(1,15)] ;
% opts.train.expDir = opts.expDir ;
opts.train.outputfea = 'true';

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    error('no datafile')
end
c = length(unique(imdb.images.labels));
d = size(imdb.images.data,2);
% Define a network similar to LeNet
f=1/100 ;

net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,1,1,50, 'single'), ...
                           'biases', zeros(1, 50, 'single'), ...
                           'stride',1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [4 1], ...
                           'stride', [2 1], ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;
                       
                       
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,1,50,40, 'single'),...
                           'biases', zeros(1,40,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [4 1], ...
                           'stride', [2 1], ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;
                       
                       
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(3,1,40,20, 'single'),...
                           'biases', zeros(1,20,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;                          
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,d,20,400, 'single'),...
                           'biases', zeros(1,400,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;    
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,400,18, 'single'),...
                           'biases', zeros(1,18,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;







% MAKE SURE the last layer's size is [1 1 X N]
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net,info, y1bag] = cnn_train(net, imdb, @getBatch, ...50
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
save('~predictions','y1bag','-v7.3');

rmpath('fcnn')

delete(['data/expnet-epoch*.mat']);

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;