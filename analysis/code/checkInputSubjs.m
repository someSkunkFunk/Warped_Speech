%checkInputSubjs
% validate script subject numbers provided -- always converts a single subj
% to subjs to facilitate looping over subjects in scripts

% also gives an experimentsBySubjNum as validation variable
if ~exist('subjs','var') && ~exist('subj', 'var')
    error('must define a subj or subjs before calling this script.')
end
if exist('subjs','var') && exist('subj', 'var')
    warning('Defined both subj and subjs vars, assuming meant subjs and deleting subjs')
    clear subj
end
if ~exist('subjs','var') || exist('subj', 'var')
    subjs=subj;
end
experimentsBySubjNum=cell(length(subjs),1);
for ss=1:length(subjs)
    subjNum=subjs(ss);
    if subjNum<23&&subjNum>0
        experimentsBySubjNum{ss}='fastSlow';
    elseif subjNum>23 &&subjNum<100
        experimentsBySubjNum{ss}='regIrreg';
    else
        error('invalidSubjs: subjs specified incorrectly.')
    end
    clear subjNum
end