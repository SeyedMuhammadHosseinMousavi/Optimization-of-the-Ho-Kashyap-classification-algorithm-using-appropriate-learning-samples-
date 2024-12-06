function [ data1,data2,data1Target,data2Target ] = ...
    MCIS( c1,c2,c1Target,c2Target,r,k )
z1=zscore(c1);
z2=zscore(c2);
[cclu1,cclu2,lblc1,lblc2]=kmean(z1,z2,k);

[d1]=pdist2(cclu1,z2,'euclidean');
[d2]=pdist2(cclu2,z1,'euclidean');

[x,temp1] = find(d2<r);
[x,temp2] = find(d1<r);

temp1=unique(temp1);
temp2=unique(temp2);

data1=c1(temp1,:);
data2=c2(temp2,:);
data1Target=c1Target(temp1,:);
data2Target=c2Target(temp2,:);
clear temp1 temp2 ;

end