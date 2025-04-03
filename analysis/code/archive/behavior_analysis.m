clear
clc
dependencies_path=('../../dependencies/');
addpath(genpath(dependencies_path));

user_profile=getenv('USERPROFILE');
for subj=2:7
    % subj = 7;
    datafolder = sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/data/',user_profile);
    
    behfile = sprintf('%ss%0.2d_WarpedSpeech.mat',datafolder,subj);
    load(behfile,'m')
    % m cols 3:6 have question 1,2 correct ans, question 1,2 subject response
    qsPerTrial=2;
    totalTrials=size(m,1);
    % filter trials w questions (every other trial)
    qI=zeros(totalTrials,1);
    qI(2:2:totalTrials)=1;
    conditions=[2/3 1 3/2]; %note maybe better to get from m itself?
    score=nan(numel(conditions),2); %nCorrect, nQuestions
    for cc=1:numel(conditions)
        cI=find(m(:,1)==conditions(cc)&qI);
        nQuestions=numel(cI)*qsPerTrial;
        tempAns=m(cI,3:end);
        nCorrect=sum((tempAns(:,1:qsPerTrial)-tempAns(:,qsPerTrial+1:end))==0,'all');
        clear tempAns
        if nCorrect<nQuestions
            score(cc,:)=[nCorrect,nQuestions];
        else
            error('wtf')
        end
    end

    fprintf('subj %d:\n',subj)
    fprintf('%d out of %d correct\n', score')
    % fprintf('%d out of %d correct\n', score(:,1),score(:,2))
end