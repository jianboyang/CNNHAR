function [xapp,xtest] = normalize(xapp,xtest)
% USAGE:
%   [xapp,xtest] = normalizemeanstd(xapp,xtest)
%   normalize inputs and output mean and standard deviation to 0 and 1
meanxapp=mean(xapp);
stdxapp=std(xapp);
[nbxapp features]=size(xapp);
for i=1:features
    if stdxapp(i)<1e-8
        stdxapp(i)=1;
    end
end
nbvar=size(xapp,2);
xapp= bsxfun(@rdivide,bsxfun(@minus,xapp, meanxapp),stdxapp) ;
if nargin >1
    nbxtest=size(xtest,1);
    xtest= bsxfun(@rdivide,bsxfun(@minus,xtest,meanxapp),stdxapp );
end;
end