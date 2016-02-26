function ypre = getypre(predictions,dataset)
k = strfind(dataset, '.mat');
if isempty(k)
    load(['data/' dataset '/xybagtst']); 
else
    load(dataset);
end

ypre_d = zeros(n,bagsize);
nbag = length(predictions);
di = 1; ii = 1;
for i = 1:nbag
    y = predictions(i);
    ypre_d(ii:ii+bagsize-1,di) = kron(y,ones(bagsize,1));
    ii = ii + oltst; di = di + 1; if di + 1 > bagsize, di = 1; end;
end
ypre = zeros(n,1);
for i = 1:n
    temp = ypre_d(i,:);
    temp(temp == 0) =  [];
    ytclass = tabulate(temp);
    if ~isempty(temp)
        [~,idx] = max(ytclass(:,2));
        ytclass = ytclass(idx,1);
        ypre(i) = ytclass;
    else
        ypre(i) = 1;
    end
end