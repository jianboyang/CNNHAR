function [y2t,y4t, y18t] = convertlabel(x)
yt = x(:,end-1:end);
l1 = yt(:,1);
l2 = yt(:,2);

glabels = [0 506616, 506617, 504616, 504617, 506620, 504620, 506605, 504605 ...
    506619, 504619, 506611, 504611, 506608, 504608, 508612, 507621, 505606];
llabels = [0, 101, 102, 104, 105];

% Null vs. Gesture
y2t = yt(:,1);
idx = find(l2 == 0);
y2t(idx) = -1*ones(length(idx),1);
idx = find(l2 ~= 0);
y2t(idx) = ones(length(idx),1);

% Locomotion
y4t = yt(:,1);
for i = 1:length(llabels)
    idx = find(l1 == llabels(i));
    y4t(idx) = i*ones(length(idx),1);
end

% Gesture
y18t = yt(:,1);
for i = 1:length(glabels)
    idx = find(l2 == glabels(i));
    y18t(idx) = i*ones(length(idx),1);
end