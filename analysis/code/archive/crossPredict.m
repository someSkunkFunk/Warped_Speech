clear, clc
user_profile=getenv('USERPROFILE');
% use these params since what we used on sfn poster and looked nicest
subj=7;
condsDir='sep_conditions/';
corrDir='corrected/';
cvDir='';
bpfilter = [1 15];

fprintf('bpfilter lims: [%g %g]\n', bpfilter(1), bpfilter(2))
ref = 'mast';
fs = 128;
datafolder = sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/data/',user_profile);
matfolder = sprintf('%smat/%g-%g_%s-ref_%dHz/%s',datafolder,bpfilter(1),bpfilter(2),ref,fs,corrDir);
matfile = sprintf('%swarpedSpeech_s%0.2d.mat',matfolder,subj);
nulldistribution_file=sprintf('%s%s%snulldistribution_s%0.2d.mat',matfolder,condsDir,cvDir,subj);

outputFile=sprintf('%s%s%smismatchPrediction_s%0.2d.mat',matfolder,condsDir, ...
    'mismatch_predict/',subj);
if ~exist(outputFile,'file')
    load(nulldistribution_file,'model_lam','stim', ...
        'resp','model','cond','r_obs','mtmin','mtmax')
    % assuming dumb cell format still in saved file:
    model=modelCell2Struct(model);
    %%
    conditions=1:3;
    r_obs_cross=zeros(3,3,size(r_obs,2));
    % train_condition, predict_condition, electrodes
    for cc=conditions
        % copy paste same-condition r values from nulldistribution file
        r_obs_cross(cc,cc,:)=r_obs(cc,:);
        for icc=conditions(conditions~=cc)
            % ccModel=model(cc);
            % not sure if crossvalidation folds make a difference in
            % mTRFpredict since models are already trained... but is then is it
            % fair to compare cross validated scores from matched conditions to
            % uncrossvalidated scores in mismatched conditions?
            [~,STATS]=mTRFpredict(stim(cond(icc)),resp(cond(icc)),model(cc));
            r_obs_cross(cc,icc,:)=STATS.r;
            clear STATS
        end
    end
    clear cc r_obs
    
    save(outputFile)
else
    fprintf('loading pre-existing file')
    load(outputFile)
end
%TODO: un-hardcode mistmatch_predict
%%
plotChn=85;
condNames={'fast','og','slow'};
xlims=[.75 3.25];
ylims=[-.09 0.12];
figure, hold on
sgtitle(sprintf('subj %d prediction accuracies - chn %d',subj,plotChn))
for cc_trf=conditions
    
    axs(cc_trf)=subplot(3,1,cc_trf);
    
    rvals=r_obs_cross(cc_trf,conditions,plotChn);

    scatter(conditions,rvals)
    
    
    title(sprintf('prediction accuracy using %s trf',condNames{cc_trf} ) )
    set(gca,'XTick',conditions, 'XTickLabel', condNames)

end
linkaxes(axs)
xlabel('experimental condition')
ylabel('rvalue')
xlim(xlims)
ylim(ylims)

hold off