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

% results.m

%% load results
clear
filedir_o='/home/decision/margro/Alcoholprediction_Groefsema/FU_AUDIT_Nov19_3';

% save was done after every iteration to be safe but we only need to load
% the very last as the array was filled up then
load([filedir_o filesep 'null_ensemble' filesep 'rep10_null2' filesep 'singlemod.mat']);
load([filedir_o filesep 'null_ensemble' filesep 'rep10_null2' filesep 'ensemble.mat']);
load([filedir_o filesep 'null_ensemble' filesep 'rep10_null2' filesep 'testset.mat']);
load([filedir_o filesep 'null_ensemble' filesep 'rep10_null2' filesep 'all_rmse.mat']);

% this is the design file including cross validation assignment that was
% used throughout
load('/home/decision/margro/Alcoholprediction_Groefsema/FU_AUDIT_Nov19_3/design.mat')
origdesign=design;
datalabels=origdesign.vars;

%%%%%%%%%%%%%%%%%%%%%%%
% fit of the ensemble %
%%%%%%%%%%%%%%%%%%%%%%%

disp(' ')
disp('overall ensemble model fit, R:')
for n=1:3
[h p(n)]=ttest2(overall_corr_test_combi(1,:,n), overall_corr_test_combi(2,:,n));
disp(sprintf('n=%d, r for ensemble is %f (p=%d), min=%f, max=%f', n, mean(overall_corr_test_combi(1,:,n)), p(n), min(overall_corr_test_combi(1,:,n)), max(overall_corr_test_combi(1,:,n))))
end
[h p]=ttest2(overall_corr_test_combi(1,:,1),overall_corr_test_combi(1,:,2));
disp(sprintf('ttest if overall_corr_tst_combi 1 and 2 are different, p=%d', p))
[h p]=ttest2(overall_corr_test_combi(1,:,1),overall_corr_test_combi(1,:,3));
disp(sprintf('ttest if overall_corr_tst_combi 1 and 3 are different, p=%d', p))

disp(' ')
disp('overall ensemble model fit, RMSE:')
for n=1:3
[h p(n)]=ttest2(overall_rmse_test(1,:,n), overall_rmse_test(2,:,n));
disp(sprintf('n=%d, r for ensemble is %f (p=%d), min=%f, max=%f', n, mean(overall_rmse_test(1,:,n)), p(n), min(overall_rmse_test(1,:,n)), max(overall_rmse_test(1,:,n))))
end
[h p]=ttest2(overall_rmse_test(1,:,1),overall_rmse_test(1,:,2));
disp(sprintf('ttest if overall_rmse_test 1 and 2 are different, p=%d', p))
[h p]=ttest2(overall_rmse_test(1,:,1),overall_rmse_test(1,:,3));
disp(sprintf('ttest if overall_rmse_test 1 and 3 are different, p=%d', p))


%%%%%%%%%%%%%%%%%%%
% single-modality %
%%%%%%%%%%%%%%%%%%%

% check if unimodal model r is significantly better than null and compare
% to ensemble
for modality=1:4
    a=r_tmp(1,:,:,modality);
    b=r_tmp(2,:,:,modality);
    c=corrtst(1,:,:,1);
    [h p(modality)]=ttest2(a(:),b(:));
    [h p2(modality)]=ttest2(a(:),c(:));
    meanz(1,modality)=mean(a(:));
    meanz(2,modality)=mean(b(:));
end
disp(' ')
disp('unimodal model fit:')
for m=1:4
    disp(sprintf('m=%d, mean r=%f, p=%d, p2=%d',m,meanz(1,m), p(m), p2(m)))
end

% check if unimodal model rmse is significantly better than null and
% compare to ensemble
for modality=1:4
    a=RMSE(1,:,:,modality);
    b=RMSE(2,:,:,modality);
    c=rmsetst(1,:,:,1);
    [h p(modality)]=ttest2(a(:),b(:));
    [h p2(modality)]=ttest2(a(:),c(:));
    meanz(1,modality)=mean(a(:));
    meanz(2,modality)=mean(b(:));
end
disp(' ')
disp('unimodal model fit:')
for m=1:4
    disp(sprintf('m=%d, mean rmse=%f, p=%d, p2=%d',m,meanz(1,m), p(m), p2(m)))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% models with each of the modalities left out %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

null=1;
for repetition=1:10
    load([filedir_o filesep 'ensemble' filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'design.mat'])
    for modality=1:4
        for cv=1:5
            usetestdata=find(design.mainfold==cv);
            set2=setdiff([1:4],modality);
            m_pred_tst{repetition,modality}(usetestdata)=[Xtst{null,repetition,cv,1}(:,set2)]*b_tmp{null,repetition,cv}(set2); %take this modality out
            m1_pred_tst{repetition,modality}(usetestdata)=[Xtst{null,repetition,cv,1}(:,modality)]*b_tmp{null,repetition,cv}(modality); %do only this modality
            a_pred_tst{repetition,modality}(usetestdata)=[Xtst{null,repetition,cv,1}(:,1:4)]*b_tmp{null,repetition,cv}(1:4); %use all
        end
        modcorr(repetition,modality)=corr(m_pred_tst{repetition,modality}',design.outcome); %take this mod out
        mod1corr(repetition,modality)=corr(m1_pred_tst{repetition,modality}',design.outcome); %use only this mod
        acorr(repetition,modality)=corr(a_pred_tst{repetition,modality}',design.outcome); %use all mods
        modrmse(repetition,modality)=sqrt(mean((m_pred_tst{repetition,modality}'-design.outcome).^2)); %take this mod out
        mod1rmse(repetition,modality)=sqrt(mean((m1_pred_tst{repetition,modality}'-design.outcome).^2)); %use only this mod
        armse(repetition,modality)=sqrt(mean((a_pred_tst{repetition,modality}'-design.outcome).^2)); %use all mods
    end
end

avgsinglemod=squeeze(mean(RMSE(1,:,:,:),3));
for modality=1:4
    [h pp(modality,1)]=ttest2(modrmse(:,modality),overall_rmse_test(1,:,1)); %is taking this mod out sig differnet
    [h pp(modality,2)]=ttest2(avgsinglemod(:,modality),overall_rmse_test(1,:,1)); %is using only this mod sig different
end
impr1=((acorr-modcorr)*100)./acorr(:,1);
impr=((acorr-mod1corr)*100)./acorr(:,1);

disp(' ')
disp('ensemble model fit wthout mod:')
for m=1:4
    disp(sprintf('m=%d, reduction in r when modality is excluded=%f pct (p=%d). reduction in r when only modality is used=%f pct (p=%d)',m,mean(impr1(:,m)),pp(m,1),mean(impr(:,m)),pp(m,2)))
end

%% check out betas
tmpusevars{1}=[1:2502]; %DID
tmpusevars{2}=[2503:3336]; %SACE
tmpusevars{3}=[3337:3614]; %struct
tmpusevars{4}=[3615:3811]; %psychometric

% get ensemble betas for all models
btmp=NaN(2,10,5,4);
for n=1:2
    for r=1:10
        for cv=1:5
            btmp(n,r,cv,:)=b_tmp{n,r,cv};
        end
    end
end


%% betas for psychometrics
pval=95;
% (1) First it was noted whether each predictor passed the significance 
% threshold in each of these 10*5 models. Predictors that were significant 
% in the majority of CV folds for each of the 10 analysis iterations were 
% said to pass the ?inner CV significance threshold?.  
for modality=1:4
    BN1PM=NaN(10,5,length(tmpusevars{modality}));
    BN2PM=NaN(10,5,length(tmpusevars{modality}));
    passPM=zeros(10,5,length(tmpusevars{modality}));
    for rep=1:10
        for cv=1:5
            BN1PM(rep,cv,:)=betas_tmp{1,rep,cv,modality}(1:length(tmpusevars{modality}));
            BN2PM(rep,cv,:)=betas_tmp{2,rep,cv,modality}(1:length(tmpusevars{modality}));
        end
    end
    [T1{1,modality},T2{1,modality},T3{1,modality},smz{1,modality}]=sigthreshs(BN1PM, BN2PM,95);
    disp(sprintf('%d T1, %d T2, %d T3, %d T4', length(smz{1,modality}{1}),length(smz{1,modality}{2}),length(smz{1,modality}{3}),length(smz{1,modality}{4})))
    [T1{2,modality},T2{2,modality},T3{2,modality},smz{2,modality}]=sigthreshs(BN1PM, BN2PM,99);
    disp(sprintf('or %d T1, %d T2, %d T3, %d T4', length(smz{2,modality}{1}),length(smz{2,modality}{2}),length(smz{2,modality}{3}),length(smz{2,modality}{4})))

    T{modality}=array2table(datalabels(tmpusevars{modality}));
    T{modality}(:,2)=array2table(squeeze(mean(mean(BN1PM,1),2)));
    T{modality}(:,3)=array2table(min(T1{1,modality}')');
    T{modality}(:,4)=array2table(sum(T1{1,modality},2));
    T{modality}(:,5)=array2table(T2{1,modality});
    T{modality}(:,6)=array2table(T3{1,modality});
    writetable(T{modality},['mod' num2str(modality) '.csv'])
end


