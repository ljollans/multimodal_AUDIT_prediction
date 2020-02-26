%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Analysis code for Groefsema, 
%  Sescousse, Luijten, Engels & Jollans 
%  2020
%
%  The following files are part of
%  the code for this project:
%  1. dataprep.m
%  2. analysis.m
%  3. results.m
%  4. sigthreshs.m
%
%  The following functions are used
%  and were downloaded from the RAFT
%  toolbox (github.com/ljollans/RAFT):
%  create_design.m
%  
%  Functions from other authors:
%  bootstrapal.m, Author: Mao Shasha

% dataprep.m

clear
%% DID
load('/home/decision/margro/ML_func_DID/MLdata_func_DID.mat')
mnDID=mask_name;
DID=meanvoldata;
DIDvolname=volume_name';
for n=1:size(DIDvolname,1)
    con1(n)=str2double(DIDvolname{n,1}{1}(length(DIDvolname{n,1}{1})-6:length(DIDvolname{n,1}{1})-4));
    Pnum1(n)=str2double((DIDvolname{n,1}{1}(length(DIDvolname{n,1}{1})-15:length(DIDvolname{n,1}{1})-13)));
end
%% SACE
load('/home/decision/margro/ML_func_SACE/MLdata_func_SACE.mat')
mnSACE=mask_name;
SACE=meanvoldata;
SACEvolname=volume_name';
clear con Pnum conPs conmasks
for n=1:size(SACEvolname,1)
    con2(n)=str2double(SACEvolname{n,1}{1}(length(SACEvolname{n,1}{1})-6:length(SACEvolname{n,1}{1})-4));
    Pnum2(n)=str2double((SACEvolname{n,1}{1}(length(SACEvolname{n,1}{1})-15:length(SACEvolname{n,1}{1})-13)));
end
%% GMV
load('/home/decision/margro/ML_struct/MLdata_struct.mat')
mnGMV=mask_name;
Struct=meanvoldata;
Structvolname=volume_name';
for n=1:size(Structvolname,1)
    Pnum3(n)=str2double(Structvolname{n,1}{1}(length(Structvolname{n,1}{1})-19:length(Structvolname{n,1}{1})-17));
end
%% merge ID lists
IDs=unique([Pnum1,Pnum2,Pnum3]); IDso=IDs;
checkIDs=zeros(length(IDs),3);
[c1 a1 b1]=intersect(Pnum1,IDs); checkIDs(b1,1)=1;
[c2 a2 b2]=intersect(Pnum2,IDs); checkIDs(b2,2)=1;
[c3 a3 b3]=intersect(Pnum3,IDs); checkIDs(b3,3)=1;
IDs=IDs(find(sum(checkIDs,2)==3));
%% reshape DID
newDID=IDs';
newDID_l={'IDs'};
u=unique(con1);
for c=1:length(u)
    f=find(con1==u(c));
    [cc a b]=intersect(newDID(:,1),Pnum1(f));
    if length(cc)==147
        newDID=[newDID,DID(:,f(b))'];
        for n=1:278
            newDID_l{length(newDID_l)+1}=['DID_' num2str(u(c)) '_' mnDID{n}];
        end
    else
        disp(length(cc))
    end
end
%% reshape SACE
[c a b]=intersect(mnDID,mnSACE);
SACE=SACE(b,:);
newSACE=IDs';
newSACE_l={'IDs'};
u=unique(con2);
for c=1:length(u)
    f=find(con2==u(c));
    [cc a b]=intersect(newSACE(:,1),Pnum2(f));
    if length(cc)==147
        newSACE=[newSACE,SACE(:,f(b))'];
        for n=1:278
            newSACE_l{length(newSACE_l)+1}=['SACE_' num2str(u(c)) '_' mnDID{n}];
        end
    else
        disp(length(cc))
    end
end
%% reshape GMV
[c a b]=intersect(IDs,Pnum3);
newGMV=IDs(a)';
newGMV=[newGMV,Struct(:,b)'];
newGMV_l={'IDs'};
for n=1:278
    newGMV_l{length(newGMV_l)+1}=['GMV_' mnDID{n}];
end
%% make one array
if isequal(newGMV(:,1),newDID(:,1)) && isequal(newGMV(:,1),newSACE(:,1))
    DATA=[newDID(:,2:end),newSACE(:,2:end),newGMV(:,2:end)];
    LABELS=[newDID_l(2:end),newSACE_l(2:end),newGMV_l(2:end)];
end
%% load psychometrics
load('/home/decision/margro/Alcoholprediction_Groefsema/newspss.mat'); X=newspss;
XX=table2array(X);
ids=table2array(X(:,1));
[c a b]=intersect(IDs,ids);
auditFU=XX(b,5);
auditBL1=XX(b,2);
load('/home/decision/margro/Alcoholprediction_Groefsema/newspssNov19.mat'); X=newspssNov19;
labels=X.Properties.VariableNames(2:end);
XX=table2array(X); 
for n=2:size(XX,2)
    ff{n}=unique(XX(:,n));
    u(n)=length(ff{n});
    for uu=1:u(n)
        t{n}(uu)=length(find(XX(:,n)==ff{n}(uu)));
        if t{n}(uu)<5 && u(n)<10
            %disp(sprintf('%d entries for %d in variable %s (%d)',t{n}(uu),ff{n}(uu),labels{n-1},n))
        end
    end
end
%recode the drug variables so it's a binary yes no 
% take out SUB9 because there are only 2 cases that have tried heroin
% sedative, cannabis, cocaine, speed, xtc, lsd, heroin
% n score>0     38   119    17    20    60    31     2
% n score>1     23   103    13    13    41    15     1
% n score>2     11    78     5     6    24     7     1
% n score>3      6    72     4     6    17     6     1
% n score>4      2    52     2     4     6     1     1
% n score>5      1    38     2     0     2     0     1
ids=table2array(X(:,1));
[c a b]=intersect(IDs,ids);
XX=XX(b,2:end); % isequal(newGMV(:,1),XX(:,1))
[f1 f2]=find(isnan(XX));
XX(f1,f2)=nanmean(XX(:,f2));

rem={'SUB9','BIS_Attention','BIS_Motor','BIS_Nonplan'};
for r=1:length(rem)
    XX(:,find(ismember(labels,rem{r})))=[];
    labels(find(ismember(labels,rem{r})))=[];
end

%% have a look at correlation structure of psychometrics
[corrs ps]=corr(XX); corrs(find(corrs==1))=NaN;
[f1,f2]=find(abs(corrs)>.9); %{'BIS_Motor'}    {'BIS_SubMotor'}
[f1,f2]=find(abs(corrs)>.85); %[labels(f1)',labels(f2)'] high correlations between individual scale items and scale totals
%% residualize out age and age2
AGE=XX(:,find(ismember(labels,'age_yr')));
XX(:,find(ismember(labels,'age_yr')))=[]; labels(find(ismember(labels,'age_yr')))=[];
DATA=zscore(DATA);
for p=1:size(DATA,2)
    aa=[[AGE,AGE.^2],(DATA(:,p))];
    mdl=fitlm(array2table(aa));
    residz(:,p)=mdl.Residuals(:,4);
end
%% make design file
auditBL=XX(:,find(ismember(labels,'Audit')));XX(:,find(ismember(labels,'Audit')))=[]; labels(find(ismember(labels,'Audit')))=[];
numreps=10; % numberof iterations of the analysis to do
data=[table2array(residz),XX];
datalabels=[LABELS,labels];
covariates=[AGE,auditBL]; % maybe add exact follow-up time and age in here?
covarlabels={'age', 'auditBL'};
outcome=auditFU';
type='linear'; 
nboot=100; 
numFolds=5; 
numLambdas=5; 
numAlphas=5; 
bagcrit='median'; 
clean=1;
saveto=['/home/decision/margro/Alcoholprediction_Groefsema' filesep 'FU_AUDIT_Nov19_3'];
subid=IDs;
balanced='balanced'; % in case your outcome is binary we will need to double check this
winsorize_data=ones(size(datalabels)); %this is for z-scoring
winsorize_data(3615:3813)=0; winsorize_data(find(ismember(datalabels,'GEN1')))=1; % i added this after the initial analysis run
winsorize_extradata=zeros(size(covarlabels)); % and z-scoring the covariates || i also changed this after the initial analysis run
exclude_binary_vars=0;

[design]=create_design(data, datalabels, covariates, covarlabels, outcome', ...
    type, nboot, numFolds, numLambdas, numAlphas, bagcrit, clean,saveto, ...
    subid, balanced, winsorize_data, winsorize_extradata, exclude_binary_vars);