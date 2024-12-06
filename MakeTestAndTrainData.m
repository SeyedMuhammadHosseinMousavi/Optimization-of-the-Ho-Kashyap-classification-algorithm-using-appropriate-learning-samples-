function [TrainData,TrainTarget,TestData,TestTarget] = MakeTestAndTrainData(List)

[row, col] = find(isnan(List));
[List,ps] = removerows(List,'ind',row);
List(List(:,end)==(min(List(:,end))),end)=0;
List(List(:,end)==(max(List(:,end))),end)=1;

X=List(:,1:end-1);
Y=List(:,end);
TrainPercent=0.7;
num_points = size(X,1);
split_point = round(num_points*TrainPercent);
seq = randperm(num_points);
TrainData = X(seq(1:split_point),:);
TrainTarget = Y(seq(1:split_point));
TestData = X(seq(split_point+1:end),:);
TestTarget = Y(seq(split_point+1:end));

end