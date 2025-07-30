clear, clc
% user_profile=getenv('USERPROFILE');
% use these params since what we used on sfn poster and looked nicest
%% compute stats_cross... or stats_cross_fair
% false does the unfair comparison we originally devised which gives
% stats_cross
fair=true;
subjs=[2:7,9:22];
% for reproducibility
rng(1);
for subj=subjs
    fprintf('starting subj %d\n',subj)
    % load saved model
    preprocess_config=config_preprocess(subj);
    do_lambda_optimization=false;
    trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
    % load preprocessed data/stimuli
    preprocessed_eeg=load(trf_config.preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
    preprocessed_eeg=preprocessed_eeg.preprocessed_eeg;
    stim=load_stim_cell(trf_config.preprocess_config,preprocessed_eeg);
    [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
        trf_config,preprocess_config);
    cond=preprocessed_eeg.cond;
    % note: idk why we didn't just save both resp and sim as col vectors to
    % begin with..
    resp=preprocessed_eeg.resp';

    

    
    % TODO: load_checkpoint is buggy and fixing it will be a pain so
    % reverting to native matlab load
    % model_data=load_checkpoint(trf_config.trf_model_path,trf_config);
    % stats_data=load_checkpoint(trf_config.model_metric_path,trf_config);
    % TODO: replace trf_config w one that has best lam after checking that both
    % checkpoint-loaded configs are correct
    if fair
        % do this janky load procedure just to get best lambda
        saved_trf_config=load(trf_config.trf_model_path,"trf_config");
        saved_trf_config=saved_trf_config.trf_config(1,1);
        best_lam=saved_trf_config.best_lam;

        shuff=randperm(numel(cond));
        cond=cond(shuff);
        resp=resp(shuff);
        stim=stim(shuff);
        % NOTE: can undo the shuffling at the end if trial-level information is
        % relevant for our statistical test by doing cond(shuff)=cond;
        n_trials=numel(cond);
        n_electrodes=size(resp{1},2);
        %initialize variables
        % each col means score from model trained on cond 1,2,3
        % test condition determined by the actual condition of the trial
        stats_cross_cv=struct('r',nan(n_trials,3,n_electrodes), ...
            'err',nan(n_trials,3,n_electrodes));

        for cc=1:3
            fprintf('%d of 3 conditions...\n',cc);
            % run LOOCV one condition at a time
            % note: some subjects missing trials so we can't assume all
            % conditions have equal number of folds
            n_folds=sum(cond==cc);
            cc_trials=find(cond==cc);
            cross_trials=find(cond~=cc);
            for k=1:n_folds
                % split condition-specific trials into train,test
                fprintf('%d of %d folds...\n',k,n_folds)
                test_trial=cc_trials(k);
                train_trials=cc_trials(cc_trials~=test_trial);
                temp_model=mTRFtrain(stim(train_trials),resp(train_trials), ...
                    preprocess_config.fs,1,trf_config.cv_tmin_ms, ...
                    trf_config.cv_tmax_ms,best_lam);

                [~,temp_stats]=mTRFpredict([stim(test_trial);stim(cross_trials)] ...
                    ,[resp(test_trial);resp(cross_trials)],temp_model);
                % TODO: check size of r works
                stats_cross_cv.r([test_trial; cross_trials],cc,:)=temp_stats.r;
                stats_cross_cv.err([test_trial; cross_trials],cc,:)=temp_stats.err;
            end 
        end
        %save to file
        save(trf_config.model_metric_path,'stats_cross_cv','-append')

    else
    
        model_data=load(trf_config.trf_model_path,"model");
        stats_data=load(trf_config.model_metric_path,"stats_obs");
    
        stats_obs=stats_data.stats_obs(2,:); clear stats_data
        model=model_data.model(2,:); clear model_data
        
        
        %%
        conditions=1:3;
        % r_cross=zeros(3,3,size(r_obs,2));
        stats_cross=cell2struct(cell(2,1),fieldnames(stats_obs));
        
        % train_condition, predict_condition, electrodes
        for cc=conditions
            % copy paste same-condition r values from nulldistribution file
            % r_cross(cc,cc,:)=stats_obs.r(cc,:);
            stats_cross(cc,cc)=stats_obs(cc); % assumes 1D struct array...
            for icc=conditions(conditions~=cc)
        
                [~,stats_cross(cc,icc)]=mTRFpredict(stim(cond==cc),resp(cond==cc),model(icc));
                % r_cross(cc,icc,:)=STATS.r;
                % clear STATS
            end
        end
        clear cc 
        fprintf('saving cross-prediction results for subj %d\n',subj)
        save(trf_config.model_metric_path,"stats_cross","-append")
        fprintf('result saved.\n')
    end
    % clear
end

%% GLMM analysis

subjs=[2:7,9:22];

disp('GLMM analysis start...')
tbl=setup_glmm_data(subjs);
% use mismatched conditions as baseline
tbl.Match=reordercats(tbl.Match,["false","true"]);
% fit GLMM
formula='Rval ~ TrainCond*Match+ (1|Subject) + (1|Subject:Electrode) + (1|DataCond)';
glme=fitglme(tbl,formula);
disp(glme)


function tbl=setup_glmm_data(subjs)
    % subjs=[2:7,9:22];
    % preallocate
    n_subjs=numel(subjs);
    rows=0;
    for subj=subjs
        tmp_pcnfg=config_preprocess(subj);
        tmp_tcnfg=config_trf(subj,false,tmp_pcnfg);
        tmp_sc=load(tmp_tcnfg.model_metric_path,"stats_cross"); % should only have 3x3 shaped vars
        tmp_sc=tmp_sc.stats_cross;
        for ii=1:3
            for jj=1:3
                [n_trials,~,n_electrodes]=size(tmp_sc(ii,jj).r);
                rows=rows+n_trials*n_electrodes;
            end
        end
    end

    Subject=categorical(repmat("",rows,1));
    DataCond=categorical(repmat("",rows,1));
    TrainCond=categorical(repmat("",rows,1));
    Match=categorical(repmat("",rows,1));
    Electrode=zeros(rows,1);
    Rval=zeros(rows,1);

    % populate table
    row=1;
    for subj=subjs
        tmp_pcnfg=config_preprocess(subj);
        tmp_tcnfg=config_trf(subj,false,tmp_pcnfg);
        tmp_sc=load(tmp_tcnfg.model_metric_path,"stats_cross"); % should only have 3x3 shaped vars
        tmp_sc=tmp_sc.stats_cross;
        for ii=1:3 % data condition
            for jj=1:3 % training condition
                R=tmp_sc(ii,jj).r;
                [n_trials,~,n_electrodes]=size(R);
                for tt=1:n_trials
                    for ee=1:n_electrodes
                        Rval(row)=R(tt,1,ee);
                        Subject(row)=categorical(subj); % does it have to be a string?
                        DataCond(row)=categorical(ii);
                        TrainCond(row)=categorical(jj);
                        Match(row)=categorical(DataCond(row)==TrainCond(row));
                        Electrode(row)=ee;
                        row=row+1;
                    end
                end
            end
        end
    end

    tbl=table(Subject,DataCond,TrainCond,Match,Electrode,Rval);
end


%%
% not sure this scatterplot adds much but should be simple to figure out...
% GLM is more pressing
% plotChn=85;
% condNames={'fast','og','slow'};
% xlims=[.75 3.25];
% ylims=[-.09 0.12];
% figure, hold on
% sgtitle(sprintf('subj %d prediction accuracies - chn %d',subj,plotChn))
% for cc_trf=conditions
% 
%     % axs(cc_trf)=subplot(3,1,cc_trf);
%     for cc_cond=conditions
%         stats_cross(conditions,cc_trf).r,plotChn);
%     end
%     scatter(conditions,rvals)
% 
% 
%     title(sprintf('prediction accuracy using %s trf',condNames{cc_trf} ) )
%     set(gca,'XTick',conditions, 'XTickLabel', condNames)
% 
% end
% linkaxes(axs)
% xlabel('experimental condition')
% ylabel('rvalue')
% xlim(xlims)
% ylim(ylims)
% 
% hold off