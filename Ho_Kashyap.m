function [test_targets] = Ho_Kashyap(train_patterns, train_targets, test_patterns)

[c, n]		   = size(train_patterns);
Max_iter=1000;b_min=0.5;eta= 0.01;
train_patterns  = [train_patterns , ones(c,1)];
train_zero      = find(train_targets == 0);
processed_patterns = train_patterns;
processed_patterns(train_zero,:) = -processed_patterns(train_zero,:);
b                  = ones(1,c);
Y                  = processed_patterns;
a                  = pinv(Y')'*b';
k	               = 0;
e    	           = 1e3;
found              = 0;
while ((sum(abs(e) > b_min)>0) && (k < Max_iter) &&(~found))
    k = k+1;
    e       = (Y * a)' - b;
    e_plus  = 0.5*(e + abs(e));
    b       = b + 2*eta*e_plus;
    a = pinv(Y')'*b';
end   %end while
test_targets =(a' * [test_patterns, ones(size(test_patterns,1),1)]')';
if (length(unique(train_targets)) == 2)
    test_targets = test_targets > 0;
end
end