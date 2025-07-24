clear, clc
% user_profile=getenv('USERPROFILE');
% use these params since what we used on sfn poster and looked nicest
subjs=[20:22];
for subj=subjs
    fprintf('starting subj %d\n',subj)
    % load saved model
    preprocess_config=config_preprocess(subj);
    do_lambda_optimization=false;
    trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
    
    % TODO: load_checkpoint is buggy and fixing it will be a pain so
    % reverting to native matlab load
    % model_data=load_checkpoint(trf_config.trf_model_path,trf_config);
    % stats_data=load_checkpoint(trf_config.model_metric_path,trf_config);
    % TODO: replace trf_config w one that has best lam after checking that both
    % checkpoint-loaded configs are correct

    model_data=load(trf_config.trf_model_path,"model");
    stats_data=load(trf_config.model_metric_path,"stats_obs");

    stats_obs=stats_data.stats_obs(2,:); clear stats_data
    model=model_data.model(2,:); clear model_data
    
    preprocessed_eeg=load(trf_config.preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
    %TODO: fix this dumb shit
    preprocessed_eeg=preprocessed_eeg.preprocessed_eeg;
    stim=load_stim_cell(trf_config.preprocess_config,preprocessed_eeg);
    [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
        trf_config,preprocess_config);
    cond=preprocessed_eeg.cond;
    resp=preprocessed_eeg.resp;
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
    % clear
end
%% GLMM

subjs=[2:7,9:22];
tbl=setup_glmm_data(subjs);

% fit GLMM
formula='Rval ~ TrainCond + (1|Subject) + (1|Subject:Electrode) + (1|DataCond)';
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
                        Electrode(row)=ee;
                        row=row+1;
                    end
                end
            end
        end
    end

    tbl=table(Subject,DataCond,TrainCond,Electrode,Rval);
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