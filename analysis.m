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

% analysis.m

%% run analysis
tmpusevars{1}=[1:2502]; %DID
tmpusevars{2}=[2503:3336]; %SACE
tmpusevars{3}=[3337:3614]; %struct
tmpusevars{4}=[3615:3812]; %psychometric

filedir_o='/home/decision/margro/Alcoholprediction_Groefsema/FU_AUDIT_Nov19_2';
mkdir(filedir_o)
mkdir([filedir_o filesep 'ensemble'])
mkdir([filedir_o filesep 'null_ensemble'])

origdesign=design;
for repetition=1:10
    for null=1:2
        if null==1
            filedir=[filedir_o filesep 'ensemble'];
        else
            filedir=[filedir_o filesep 'null_ensemble'];
        end
        %CV assignment for null,repetition
        design=origdesign;
        zz=shuffle([1:length(design.mainfold)]);
        design.mainfold=design.mainfold(zz);
        design.subfolds=design.subfolds(zz,:);
        if null==2
            Z=[1:length(design.outcome)]; Z=shuffle(Z);
            design.data=design.data(Z,:);
            design.extradata=design.extradata(Z,:);
        end
        mkdir([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)])
        save([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'design.mat'],'design');
        
        for cv=1:5
            usedata=find(design.mainfold~=cv);
            truth=design.outcome(usedata);
            mf=design.subfolds(usedata,cv);
            
            %% individual modality models
            for modality=1:4
                usevars=tmpusevars{modality};
                X=[design.data(usedata,usevars),design.extradata(usedata,:)];
                kb=[ones(size(X,1),1), X];
                clear btmp pred
                for boot=1:100
                    parfor n=1:5
                        ftrain=find(mf~=n);
                        [Xboot,Yboot,indexselect]=bootstrapal(kb(ftrain,:),truth(ftrain),2/3);
                        [btmp(boot,:,n),bint,r,rint,stats] = regress(Yboot,Xboot);
                    end
                end
                b=squeeze(mean(btmp,1));
                for n=1:5
                    ftest=find(mf==n);
                    pred(ftest)=glmval([b(:,n)],kb(ftest,2:end), 'identity');
                end
                [r_tmp(null,repetition,cv,modality) , p]=corr(pred', truth);
                RMSE(null,repetition,cv,modality) = sqrt(mean((truth-pred').^2));
                betas_tmp{null,repetition,cv,modality}=mean(b(2:end,:),2);
                predval_tmp{null,repetition,cv}(:,modality)=pred;
            end
            
            %% ensemble model
            X=predval_tmp{null,repetition,cv};
            kb=[ones(size(X,1),1), X];
            clear btmp pred
            for boot=1:100
                parfor n=1:5
                    ftrain=find(mf~=n);
                    [Xboot,Yboot,indexselect]=bootstrapal(kb(ftrain,:),truth(ftrain),2/3);
                    [btmp(boot,:,n),bint,r,rint,stats] = regress(Yboot,Xboot);
                end
            end
            b=squeeze(mean(btmp,1));
            for n=1:5
                ftest=find(mf==n);
                pred(ftest)=glmval([b(:,n)],kb(ftest,2:end), 'identity');
            end
            [rs_tmp(null,repetition,cv) , ps_tmp(null,repetition,cv)]=corr(pred', truth);
            mses_tmp(null,repetition,cv)=(abs(truth-pred')'*abs(truth-pred')/length(truth));
            RMSEs_tmp(null,repetition,cv)=sqrt(mean((truth-pred').^2));
            b_tmp{null,repetition,cv}=mean(b(2:end,:),2);
            pred_tmp{null,repetition,cv}=pred;
            
            %% calculate test set values
            usetestdata=find(design.mainfold==cv);
            for modality=1:4
                usevars=tmpusevars{modality};
                tmp_b1=betas_tmp{null,repetition,cv,modality};
                Xtst{null,repetition,cv,1}(:,modality)=[design.data(usetestdata,usevars),design.extradata(usetestdata,:)]*tmp_b1;
                Xtst{null,repetition,cv,2}(:,modality)=[design.extradata(usetestdata,:)]*tmp_b1([length(tmp_b1)-1:length(tmp_b1)]);
                Xtst{null,repetition,cv,3}(:,modality)=[design.data(usetestdata,usevars)]*tmp_b1([1:length(tmp_b1)-2]);
            end
            pred_tst{null,repetition,cv,1}=[Xtst{null,repetition,cv,1}]*b_tmp{null,repetition,cv};
            corrtst(null,repetition,cv,1)=corr(pred_tst{null,repetition,cv,1},design.outcome(usetestdata));
            shrinkage(null,repetition,cv)=rs_tmp(null,repetition,cv)-corrtst(null,repetition,cv);
            allpred_test{null,repetition,1}(usetestdata)=pred_tst{null,repetition,cv,1};
            
            pred_tst{null,repetition,cv,2}=[Xtst{null,repetition,cv,2}]*b_tmp{null,repetition,cv};
            corrtst(null,repetition,cv,2)=corr(pred_tst{null,repetition,cv,2},design.outcome(usetestdata));
            allpred_test{null,repetition,2}(usetestdata)=pred_tst{null,repetition,cv,2};
            
            pred_tst{null,repetition,cv,3}=[Xtst{null,repetition,cv,3}]*b_tmp{null,repetition,cv};
            corrtst(null,repetition,cv,3)=corr(pred_tst{null,repetition,cv,3},design.outcome(usetestdata));
            allpred_test{null,repetition,3}(usetestdata)=pred_tst{null,repetition,cv,3};
        end
        overall_corr_test_combi(null,repetition,1)=corr(allpred_test{null,repetition,1}',design.outcome);
        overall_corr_test_combi(null,repetition,2)=corr(allpred_test{null,repetition,2}',design.outcome);
        overall_corr_test_combi(null,repetition,3)=corr(allpred_test{null,repetition,3}',design.outcome);
        save([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'singlemod.mat'],'r_tmp','betas_tmp', 'predval_tmp');
        save([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'ensemble.mat'],'rs_tmp','b_tmp', 'pred_tmp', 'mses_tmp');
        save([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'testset.mat'],'Xtst','pred_tst','corrtst','allpred_test','shrinkage', 'overall_corr_test_combi');
    end
end

%% went back in to also calculate RMSE
clear
saveto=['/home/decision/margro/Alcoholprediction_Groefsema' filesep 'FU_AUDIT_Nov19_3'];
load([saveto filesep 'design.mat'])
origdesign=design;

tmpusevars{1}=[1:2502]; %DID
tmpusevars{2}=[2503:3336]; %SACE
tmpusevars{3}=[3337:3614]; %struct
tmpusevars{4}=[3615:3812]; %psychometric

filedir_o='/home/decision/margro/Alcoholprediction_Groefsema/FU_AUDIT_Nov19_3';

for repetition=1:10
    for null=1:2
        if null==1
            filedir=[filedir_o filesep 'ensemble'];
        else
            filedir=[filedir_o filesep 'null_ensemble'];
        end
        %CV assignment for null,repetition
        load([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'design.mat']);
        
        load([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'singlemod.mat']);
        %r_tmp, betas_tmp, predval_tmp
        load([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'ensemble.mat']);
        %rs_tmp, b_tmp, pred_tmp, mses_tmp
        load([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'testset.mat'])
        %,'Xtst','pred_tst','corrtst','allpred_test','shrinkage', 'overall_corr_test_combi'
        
        for cv=1:5
            usedata=find(design.mainfold~=cv);
            truth=design.outcome(usedata);
            mf=design.subfolds(usedata,cv);
            
            %% individual modality models
            for modality=1:4
                pred=predval_tmp{null,repetition,cv}(:,modality)';
                if isequal(r_tmp(null,repetition,cv,modality),corr(pred', truth))
                    RMSE(null,repetition,cv,modality) = sqrt(mean((truth-pred').^2));
                else
                    pause
                end
            end
            
            %% ensemble model
            pred=pred_tmp{null,repetition,cv};
            if isequal(rs_tmp(null,repetition,cv) , corr(pred', truth))
                RMSEs_tmp(null,repetition,cv)=sqrt(mean((truth-pred').^2));
            else
                pause
            end
            
            %% calculate test set values
            usetestdata=find(design.mainfold==cv);
            % variations are: (1) data + covars, (2) just covars, (3) just data
            for variation=1:3
                if isequal(corrtst(null,repetition,cv,variation),corr(pred_tst{null,repetition,cv,variation},design.outcome(usetestdata)))
                    rmsetst(null,repetition,cv,variation)=sqrt(mean((design.outcome(usetestdata)-pred_tst{null,repetition,cv,variation}).^2));
                else
                    pause
                end
            end
        end
        for variation=1:3
            if isequal(overall_corr_test_combi(null,repetition,variation),corr(allpred_test{null,repetition,variation}',design.outcome))
                overall_rmse_test(null,repetition,variation)=sqrt(mean((design.outcome'-allpred_test{null,repetition,variation}).^2));
            else
                pause
            end
        end
        
        save([filedir filesep 'rep' num2str(repetition) '_null' num2str(null)  filesep 'all_rmse.mat'],'RMSE','RMSEs_tmp','overall_rmse_test', 'rmsetst');
    end
end