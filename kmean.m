function [cclu1,cclu2,lblc1,lblc2]=kmean(c1,c2,x)

[lblc1,cclu1]=kmeans(c1,x);
[lblc2,cclu2]=kmeans(c2,x);

end