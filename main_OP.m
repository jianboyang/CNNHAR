clc
clear
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s)


dataset = 'S1_label18'; wins = 100;


load(['data/' dataset '/xybagtst']);
for m = 1
    switch m
        case 1
            method = 'cnn';
            y1bag = fcnn(dataset);
            y1 = getypre(y1bag,dataset);
        case 2
            method = 'cnn_smoothing';
            resultfile = ['result/' dataset '_cnn'];
            y1 = fsmoothing(resultfile, wins);
    end
    % Calculate the reulst
    [acc, af, nf] = Results_statistics (ytst, y1);
    fprintf(['acc = %f, af = %f, nf = %f\n'],100*acc, 100*af, 100*nf);
    save(['result/' dataset '_' method],'acc','af','nf','y1')
    C = confusionmat(ytst,y1);
end
