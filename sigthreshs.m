function [A,B,C,summary]=sigthreshs(betasactual, betasnull,pval)
nrep=size(betasactual,1); 
ncv=size(betasactual,2);
%disp(sprintf('%d repetitions, %d CV folds', nrep,ncv))

% 1
A=zeros(size(betasactual,3),nrep);
for r=1:nrep
    tmp=zeros(size(betasactual,3),ncv);
    for cv=1:ncv
        tmp(find(abs(betasactual(r,cv,:))>prctile(abs(betasnull(r,cv,:)),pval)),cv)=1;
    end
    A(:,r)=sum(tmp,2);
end
summary{1}=find(min(A')>0);

% 2
B=zeros(size(betasactual,3),nrep);
for r=1:nrep
    B(find(abs(squeeze(mean(betasactual(r,:,:),2)))>prctile(abs(squeeze(mean(betasnull(r,:,:),2))),pval)),r)=1;
end
B=sum(B,2);
summary{2}=find(B>4);

%3
C=zeros(size(betasactual,3),1);
C(find(abs(mean(mean(betasactual,1),2))>prctile(abs(mean(mean(betasnull,1),2)),pval)))=1;
summary{3}=find(C>0);

% smz
[c a b]=intersect(summary{1},summary{2});
nn=c;
[c a b]=intersect(nn,summary{3});
summary{4}=c;