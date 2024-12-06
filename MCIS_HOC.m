clear all ; close all hidden; clc ;
load('haberman');
cump1=0;
mt1=0;
cump2=0;
mt2=0;
itr=10;
% for k=1:10
% for r=1:0.5:10
for l=1:itr;
%     p1=0;p2=0;
    %% make Train And Test
    [TrainData,TrainTarget,TestData,TestTarget] =...
        MakeTestAndTrainData(haberman);
    
    % c1=0
    c1=TrainData(TrainTarget(:,1)==0,:);
    c1Target=TrainTarget(TrainTarget(:,1)==0,:);
    %c2=1
    c2=TrainData(TrainTarget(:,1)==1,:);
    c2Target=TrainTarget(TrainTarget(:,1)==1,:);
    
    %Permute TestData
    TestOut=TestTarget;
    TestIn=TestData;
    
    %% MCIS:
    % for haberman k=2 r=3 3.5 , k=4 r=1 , k=5 r=1.5 , k=10 r=1.5
    
    % for pimaindiansdiabetes k=2 r=4.5 , k=8 r=3.5 , k=10 r=4

    % for ionosphere k=2 r=9 , k=5 r=7.5 , k=8 r=7.5

    % for breastcancerwisconsin k=2 r=4 6 , k=3 r=3.5 , k=4 r=6.5 ,
    %                           k=5 r=3 3.5 , k=7 r=3 3.5 4 5 5.5 ,
    %                           k=10 r=5.5 6.5
%     
    k=4; %Number of cluster
    r=1; % Radios of nearest cluster
    
    [data1,data2,data1Target,data2Target]=MCIS(c1,c2,c1Target,c2Target,r,k);
    
    %% My Ho_Kashyap
    
    train_patterns=[data1;data2];
    train_targets=[data1Target;data2Target];
    test_patterns=TestIn;
    tic
    [test_targets]=Ho_Kashyap(train_patterns,train_targets,test_patterns);
    t1=toc;
    [c,cm,ind,per]=confusion(TestOut',test_targets');
    
    p1=1-c;
    cump1=cump1+p1;
    mt1=mt1+t1;
    
    %% Real Ho_Kashyap
    
    train_patterns=[c1;c2];
    train_targets=[c1Target;c2Target];
    test_patterns=TestIn;
    tic
    [Real_test_targets]=Ho_Kashyap(train_patterns,train_targets,test_patterns);
    t2=toc;
    [cn2,cm,ind,per]=confusion(TestOut',Real_test_targets');
    
    p2=1-cn2;
    cump2=cump2+p2;
    mt2=mt2+t2;
end

percent1=(cump1/itr)*100;
percent2=(cump2/itr)*100;

% if(percent1-percent2>2)
    disp([k r percent1 mt1/itr percent2 mt2/itr]);
% end
% percent1=0;mt1=0;cump1=0;
% percent2=0;mt2=0;cump2=0;
% end %end k
% end %end r
clear ans mt1 mt2 t1 t2 c cm cn2 cump1 cump2 ind ...
    itr k l p1 p2 r percent1 percent2 per test_patterns ...
    TestIn TestOut train_patterns train_targets;