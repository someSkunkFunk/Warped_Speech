clear, clc
% user_profile=getenv('USERPROFILE');
% use these params since what we used on sfn poster and looked nicest
%NOTE: code currently ignores fact that at least one subject has a missing
%trial
%% compute stats_cross... or stats_cross_fair
% false does the unfair comparison we originally devised which gives
% stats_cross - true crossvalidates so that all scores are based on unseen
% data
compute_stats_cross=false;
overwrite_stats_cross=false;
if compute_stats_cross
    fair=true;
    % trim end of trials for og/slow to match number of samples in fast trials
    trim_stimuli=true;
    %todo: include an override variable
    subjs=[2:7,9:22];
    % for reproducibility
    rng(1);
    for subj=subjs
        fprintf('starting subj %d\n',subj)
        % load saved model
        preprocess_config=config_preprocess(subj);
        do_lambda_optimization=false;
        trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
        if ~any(ismember({"stats_cross","stats_cross_cv"}, ...
                who("-file",trf_config.model_metric_path)))||overwrite_stats_cross
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
            if trim_stimuli
                min_ns=min(cellfun('size',stim,1));
                stim=cellfun(@(x) x(1:min_ns),stim,'UniformOutput',false);
                resp=cellfun(@(x) x(1:min_ns,:),resp,'UniformOutput',false);
            end
        
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
                stats_cross_cv=struct('r',nan(n_trials,n_trials,n_electrodes), ...
                    'err',nan(n_trials,n_trials,n_electrodes));
        
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
                        
                        stats_cross_cv.r(test_trial,test_trial,:)=temp_stats.r(1,:);
                        stats_cross_cv.err(test_trial,test_trial,:,:)=temp_stats.err(1,:);
        
                        stats_cross_cv.r(test_trial,cross_trials,:)=temp_stats.r(2:end,:);
                        stats_cross_cv.err(test_trial,cross_trials,:)=temp_stats.err(2:end,:);
                    end 
                end
                %NOTE: need to inverst shuffling before saving!
                % that way their indices correspond with the original order that's saved
                % in preprocess_config
                stats_cross_cv.r(shuff,shuff,:)=stats_cross_cv.r;
                stats_cross_cv.err(shuff,shuff,:)=stats_cross_cv.err;
                %save to file
                save(trf_config.model_metric_path,'stats_cross_cv','-append')
        
            else
                % this comparison didn't hold out data so is unfair
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
        end
        % clear
    end
end
%% setup r-values for stats

get_rcross=true;
avg_cross_trials_rcross=true;
% note: not sure how to interpret non-averaged results so code really just
% assumes this is true...
overwrite_Rcs=false;

if get_rcross 
    subjs=[2:7,9:22];
    % note: can't figure out how to pre-a
    n_electrodes=128; % how to avoid hardcoding? does it even matter?
    n_cond=3;
    n_subjs=numel(subjs);
    all_subj_Rcs=nan(n_subjs,n_cond,n_cond,n_electrodes);
    % subjs=2;
    do_lambda_optimization=false;
    for ss=1:numel(subjs)
        subj=subjs(ss);
        % load saved model
        fprintf('fetching subj %d Rcs...\n',subj);
        preprocess_config=config_preprocess(subj);
                trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
        clear preprocess_config
        if ~ismember("Rcs",who("-file",trf_config.model_metric_path))||overwrite_Rcs
            fprintf('compiling subj %d Rcs...\n',subj);
            % load preprocessed data/stimuli
            preprocessed_eeg=load(trf_config.preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
            preprocessed_eeg=preprocessed_eeg.preprocessed_eeg;
            cond=preprocessed_eeg.cond;
            clear preprocessed_eeg
            stats_cross_cv=load(trf_config.model_metric_path,"stats_cross_cv");
            stats_cross_cv=stats_cross_cv.stats_cross_cv;
            %%
            % [R_ff,R_fo,R_fs,...
            % R_of, R_oo, R_os,...
            % R_sf,R_so,R_ss]
            Rcs=compile_rvals(stats_cross_cv,cond,avg_cross_trials_rcross);
            fprintf('saving Rcs for subj %d...\n',subj);
            save(trf_config.model_metric_path,"Rcs","-append")
        else
            fprintf('loading from saved file...\n')
            load(trf_config.model_metric_path,"Rcs")
        end
        clear trf_config
        all_subj_Rcs(ss,:,:,:)=Rcs;
        
        
    end
    
end

cond={'fast','og','slow'};
% arrange r_cross/r_within
[R_within,R_cross]=split_all_subj_Rs(all_subj_Rcs);
% [subj X train cond X electrodes],[subj X train cond X test cond X electrodes]

%% load chanlocs
global boxdir_mine
loc_file=sprintf("%s/analysis/128chanlocs.mat",boxdir_mine);
load(loc_file);

%% oostenveld permutation test of within vs cross prediction accuracies

%% generate permuted distribution of contrast maps
% note: some permutations are redundant with this contras calculation so we
% could make it more efficient by ignoring those...
D_obs=R_within-squeeze(mean(R_cross,3));
n_perm=6000;
[~,perm_idx]=sort(rand(n_perm,n_subjs,n_cond,n_cond),4);
% n_perm independent permutations per subject per condition
D_perm=nan(n_perm,n_subjs,n_cond,n_electrodes);
for nn=1:n_perm
    for ss=1:n_subjs
        for cc=1:n_cond
            within_perm_idx=squeeze(perm_idx(nn,ss,cc,1));
            cross_perm_idx=squeeze(perm_idx(nn,ss,cc,2:n_cond));
            S_within_=squeeze(all_subj_Rcs(ss,within_perm_idx,cc,:));
            S_cross_=squeeze(mean(all_subj_Rcs(ss,cross_perm_idx,cc,:),2));
            D_perm(nn,ss,cc,:)=S_within_-S_cross_;
            clear S_within_ S_cross_
        end
    end
end

%% compute group-level statistics
% note: alpha is 0.05 and since t-dist is symmetric about zero can just
% calculate cdf^-1 up to alpha and abs it 
crit_p=0.05;
t_thresh=abs(tinv(crit_p,n_subjs-1));
% get one-sample t-statistic of mean contrasts
T_perm=squeeze(mean(D_perm,2)./(std(D_perm,0,2)/sqrt(n_subjs)));
T_obs=squeeze(mean(D_obs,1)./(std(D_obs,0,1)/sqrt(n_subjs)));
%% plot D_obs & T_obs topos
for cc=1:n_cond
    for ss=1:n_subjs
        figure
        topoplot(D_obs(ss,cc,:),chanlocs);
        colorbar
        title(sprintf('subj %d R_{within}-mean(R_{cross}) - %s',...
            subjs(ss),cond{cc}))
    end
end

for cc=1:n_cond
    figure
    topoplot(T_obs(cc,:),chanlocs);
    colorbar
    title(sprintf('t-statistic for %s',cond{cc}))
end
%% scatter plots D_obs & T_obs
for ss=1:n_subjs
    figure
    for cc=1:n_cond
        scatter(repmat(cc,n_electrodes,1),squeeze(D_obs(ss,cc,:)));
        hold on
    end
    title(sprintf('subj %d - all electrodes',subjs(ss)))
    xlabel('condition')
    ylabel('R_{within}-mean(R_{cross})')
    xticks(1:n_cond)
    xticklabels(cond)
    hold off
end
figure
for cc=1:n_cond
    scatter(repmat(cc,n_electrodes,1),squeeze(T_obs(cc,:)));
    hold on
end
title(sprintf('across-subjects t-statistic - all electrodes'))
xlabel('condition')
ylabel('tstat(R_{within}-mean(R_{cross}))')
xticks(1:3)
xticklabels(cond)
hold off
%% Scatterplot R_within and R_Cross


%% across all subjects
% rwithin
figure
for cc=1:n_cond
    R_=squeeze(R_within(:,cc,:));
    R_=R_(:);
    scatter(repmat(cc,n_electrodes*n_subjs,1),R_);
    hold on
    clear R_
end
title('all subj & all electrodes')
xlabel('condition')
ylabel('R_{within}')
xticks(1:n_cond)
xticklabels(cond)
hold off
%rcross
% get cross condition pairing labels
pairs_=get_off_diag_pairs(n_cond);
% sort them by test condition
[~,Ipairs]=sort(pairs_(:,2));
pairs_=pairs_(Ipairs,:);
n_cross=length(pairs_);

cross_ids=cell(size(pairs_,1),1);
for pp=1:n_cross
    cross_ids{pp}=sprintf('%s,%s',cond{pairs_(pp,1)},cond{pairs_(pp,2)});
end
figure
for pp=1:n_cross
    R_=squeeze(R_cross(:,pairs_(cc,1),pairs_(cc,2),:));
    R_=R_(:);
    scatter(repmat(pp,n_electrodes*n_subjs,1),R_);
    hold on
end

title('all subjs & all electrodes')
xlabel('train->test')
ylabel('R_{cross}')
xticks(1:n_cross)
xticklabels(cross_ids)
hold off
%% subject-level
% rwithin
figure
for ss=1:n_subjs
    for cc=1:n_cond
        R_=squeeze(R_within(:,cc,:));
        R_=R_(:);
        scatter(repmat(cc,n_electrodes*n_subjs,1),R_);
        hold on
        clear R_
    end
    title(sprintf('subj %d & all electrodes',subjs(ss)))
    xlabel('condition')
    ylabel('R_{within}')
    xticks(1:n_cond)
    xticklabels(cond)
    hold off
end

%rcross
for ss=1:n_subjs
    for pp=1:n_cross
        cross_ids{pp}=sprintf('%s,%s',cond{pairs_(pp,1)},cond{pairs_(pp,2)});
    end
    figure
    for pp=1:n_cross
        R_=squeeze(R_cross(:,pairs_(cc,1),pairs_(cc,2),:));
        R_=R_(:);
        scatter(repmat(pp,n_electrodes*n_subjs,1),R_);
        hold on
    end
    
    title(sprintf('subjs %d & all electrodes',subjs(ss)))
    xlabel('train->test')
    ylabel('R_{cross}')
    xticks(1:n_cross)
    xticklabels(cross_ids)
    hold off
end
%% define electrode neighborhoods

% not this will affect clustering results
adj=get_adjacency_mat(chanlocs);
vis_neighbors=true; % note this will generate 129 figures!
% having trouble keeping head size consistent when calling topoplot
% consecutively... opting instead to have each neighborhood plotted
% separately
figure
topoplot([],chanlocs,'electrodes','on','style','blank','headrad',.5);
title('electrode positions')
if vis_neighbors
    % check adjacency matrix
    for chn_=1:n_electrodes
        figure
        topoplot([],chanlocs,'electrodes','on','style','blank','plotchans', ...
            find(adj(chn_,:)),'emarker',{'o','b',10,1},'headrad',.5);
        title(sprintf('%s + neighbors',chanlocs(chn_).labels))
        % topoplot([],chanlocs(chn_),'conv','off','numcontour',0);
        % plot(X_(chn_),Y_(chn_),chanlocs)
        % hold off
    end
    clear chn_
end

%% run cluster analysis
% cluster analysis on T_obs
obs_clusters=cell(3,1);obs_masses=cell(3,1);
for cc=1:n_cond
    [obs_clusters{cc}, obs_masses{cc}]=clusterize(T_obs(cc,:),t_thresh,adj);
end
% do cluster test on T_perm
cluster_nulldist=get_cluster_nulldist(T_perm,t_thresh,adj);
%% plot nulldistributions of oostenveld cluster analysis

%plot perm distributions for each condition
crit_mass=nan(n_cond,1);
for cc=1:n_cond
    figure
    [F_null_,Fmass_null_]=ecdf(cluster_nulldist(:,cc));
    %note: need to call twice to plot? that's outrageous
    ecdf(cluster_nulldist(:,cc));
    hold on
    % get critical mass
    crit_mass(cc)=min(Fmass_null_(F_null_>(1-crit_p)));
    plot(repmat(crit_mass(cc),2,1), [0 1], 'r--')
    xlabel('cluster mass')
    ylabel('probability')
    legend('cdf',sprintf('crit mass: %0.1f',crit_mass(cc)))
    title(sprintf('%s null',cond{cc}))
    hold off
    

    clear Fmass_null_ F_null_
end

%% topoplots above-threshold clusters
for cc=1:n_cond
    
    
    clusts_=obs_clusters{cc};
    masses_=obs_masses{cc};
    sig_clusts_=masses_>crit_mass(cc);

    
    fprintf('for %s, any clusters observed above %0.2f threshold? %d\n', ...
        cond{cc},crit_p,...
        any(sig_clusts_))
    if any(sig_clusts_)
        for sc=1:numel(sig_clusts_)
            figure
            % clust_ts_=zeros(n_electrodes,1);
            % clust_ts_(clusts_{sc})=T_obs(cc,clusts_{sc});
            % 
            % topoplot(clust_ts_,chanlocs)
            %just show locations
            topoplot([],chanlocs,'electrodes','on','style','blank', ...
                'plotchans', clusts_{sc},'emarker',{'o','b',10,1});
            title(sprintf('test cond:%s, cluster %d of %d',cond{cc},sc,numel(sig_clusts_)))
            clear clust_ts

        end

    end

    clear clusts_ masses_ sig_clusts_
end
%% investigate if observed clusters actually disjoint
% note: we determined they are disjoint, just have weird morphologies due
% to how we've defined neighbors
% [clusters_disjoint_,cluster_intersections_]=check_disjoint(obs_clusters{1});
%% percent change, rcross/rwithin, bootsrapping analyses (old)
% %% generate r cross vs r within proportions/%change
% R_within_expanded_=permute(repmat(R_within,1,1,1,n_cond-1),[1,2,4,3]);
% R_cross_div_w=R_cross./R_within_expanded_;
% R_percent_change=100.*(R_cross-R_within_expanded_)./R_within_expanded_;
% clear R_within_expanded_
% 
% n_subjs=size(R_cross_div_w,1);
% pairs_=get_off_diag_pairs(n_cond);
% %sort them by train condition
% [~,Ipairs]=sort(pairs_(:,1));
% pairs_=pairs_(Ipairs,:);
% 
% comparison_ids=cell(size(pairs_,1),1);
% for pp=1:size(comparison_ids,1)
%     comparison_ids{pp}=sprintf('%s,%s',cond{pairs_(pp,1)},cond{pairs_(pp,2)});
% end
% %% plot r_cross/r_within scatters
% % is there a good way to show data from multiple electrodes?
% plot_electrode=85; 
% 
% figure
% hold on
% loop_=0; % dummy counter
% for cc=1:n_cond
%     scatter((loop_+1)+zeros(n_subjs,1),R_cross_div_w(:,cc,1,plot_electrode))
%     scatter((loop_+2)+zeros(n_subjs,1),R_cross_div_w(:,cc,2,plot_electrode))
%     loop_=loop_+2;
% end
% clear loop_
% set(gca(),'XTick',1:size(comparison_ids,1),'XTickLabels',comparison_ids)
% ylabel('R_{cross}/R_{within}')
% xlabel('Train condition -> test condition')
% title(sprintf('electrode: %d',plot_electrode))
% hold off
% %% plot % change scatters
% figure
% hold on
% loop_=0; % dummy counter
% for cc=1:n_cond
%     scatter((loop_+1)+zeros(n_subjs,1),R_percent_change(:,cc,1,plot_electrode))
%     scatter((loop_+2)+zeros(n_subjs,1),R_percent_change(:,cc,2,plot_electrode))
%     loop_=loop_+2;
% end
% clear loop_
% set(gca(),'XTick',1:size(comparison_ids,1),'XTickLabels',comparison_ids)
% ylabel('%change(R_{cross},R_{within})')
% xlabel('Train condition -> test condition')
% title(sprintf('electrode: %d',plot_electrode))
% hold off
% %% histograms of r_cross/r_within
% figure
% loop_=1; % dummy counter
% for cc_tr=1:n_cond
%     for cc_te=1:n_cond-1
%         subplot(n_cond,n_cond-1,loop_)
%         histogram(R_cross_div_w(:,cc_tr,cc_te,plot_electrode))
%         xlabel('R_{cross}/R_{within}')
%         title(sprintf('train -> test %s - electrode: %d', ...
%             comparison_ids{loop_},plot_electrode))
%         loop_=loop_+1;
% 
%     end
% end
% clear loop_
% %% histograms of percent change
% figure
% loop_=1; % dummy counter
% for cc_tr=1:n_cond
%     for cc_te=1:n_cond-1
%         subplot(n_cond,n_cond-1,loop_)
%         histogram(R_percent_change(:,cc_tr,cc_te,plot_electrode))
%         xlabel('100*(R_{cross}-R_{within})/R_{within}')
%         title(sprintf('train -> test %s - electrode: %d', ...
%             comparison_ids{loop_},plot_electrode))
%         loop_=loop_+1;
%     end
% end
% clear loop_
% 
% %% plot individual subject topos of percent change
% clims_=[-200,200];
% for ss=1:n_subjs
%     figure
%     p_=1;
%     for cc_tr=1:n_cond
%         for cc_te=1:n_cond-1
%             subplot(n_cond,n_cond-1,p_)
%             topoplot(R_percent_change(ss,cc_tr,cc_te,:),chanlocs);
%             title(sprintf('train,test: %s',comparison_ids{p_}))
%             clim(clims_)
%             colorbar
%             p_=p_+1;
%         end
%     end
%     sgtitle(sprintf('Subj %d %%change(r_{within},r_{cross})',subjs(ss)))
%     clear p_ clims_
% end
% 
% %% plot subj-averaged topos of percent change
% R_percent_change_mean=mean(R_percent_change,1);
% figure
% p_=1;
% clims_=[-200,200];
% for cc_tr=1:n_cond
%     for cc_te=1:n_cond-1
%         subplot(n_cond,n_cond-1,p_)
%         topoplot(R_percent_change_mean(:,cc_tr,cc_te,:),chanlocs);
%         title(sprintf('train,test: %s',comparison_ids{p_}))
%         clim(clims_)
%         colorbar
%         p_=p_+1;
%     end
% end
% sgtitle('subj-avg  %change(r_{cross},r_{within})')
% clear p_ clims_
% 
% %% bootstrap %-change
% %
% boot_N=1000; % number of bootstrap samples
% R_booted_percent_change=R_percent_change(randi(n_subjs,[boot_N,n_subjs]),:,:,:);
% R_booted_percent_change=reshape(R_booted_percent_change,n_subjs,boot_N,n_cond,n_cond-1,n_electrodes);
% % average across subjects in each sample
% R_booted_percent_change=squeeze(mean(R_booted_percent_change,1));
% %% get quantiles [2.5%,97.5%]
% qs=[.025,0.975];
% boot_qtls=quantile(R_booted_percent_change,qs);
% 
% %% histogram bootstrap results
% figure
% loop_=1; % dummy counter
% ylims_=[0 0.2];
% for cc_tr=1:n_cond
%     for cc_te=1:n_cond-1
%         subplot(n_cond,n_cond-1,loop_)
%         histogram(R_booted_percent_change(:,cc_tr,cc_te,plot_electrode), ...
%             'Normalization','probability')
%         hold on
%         % add line at 95% CIs
%         xci_=boot_qtls(:,cc_tr,cc_te,plot_electrode);
%         yci_=repmat(ylims_',1,numel(xci_));
%         xci_=[xci_,xci_];
%         plot(xci_',yci_,'r--')
%         xlabel('mean(100*(R_{cross}-R_{within})/R_{within})')
%         title(sprintf('train -> test %s - electrode: %d', ...
%             comparison_ids{loop_},plot_electrode))
%         ylim(ylims_)
%         loop_=loop_+1;
%     end
% end
% sgtitle('Bootstrap % change')
% clear loop_ ylims_ xci_ yci_
% %% topoplot of CI results
% % plot which electrodes have improved/impoverished prediction accuracy as
% % determined by 0 being contained within the 95% CI
% figure
% p_=1;
% clims_=[-1 1];
% for cc_tr=1:n_cond
%     for cc_te=1:n_cond-1
%         boot_ci_topo_=sign(squeeze(boot_qtls(:,cc_tr,cc_te,:)));
%         boot_ci_topo_=sign(sum(boot_ci_topo_));
%         subplot(n_cond,n_cond-1,p_)
%         topoplot(boot_ci_topo_,chanlocs);
%         title(sprintf('train,test: %s',comparison_ids{p_}))
%         clim(clims_)
%         colorbar
%         p_=p_+1;
%     end
% end
% sgtitle('bootstrapped subj-avg %change(r_{cross},r_{within})')
% clear p_ boot_ci_topo_ clims_
%% Welch t-test (old)
% % unshuffle conditions - Note: will just have to re-load them when
% % iterating over all subjects
% run_welchtest=false;
% overwrite_welch=false;
% % note: welch test assumes datapoints are independent... but cross trials
% % are tested 25 times on the same trial for each train condition, so aren't
% % really independent... averaging out might be the best way to deal with
% % that
% avg_cross_trials_rcross=true;
% if run_welchtest 
%     subjs=[2:7,9:22];
%     for subj=subjs
%         % load saved model
%         fprintf('subj %d Welch ttest...\n',subj);
%         preprocess_config=config_preprocess(subj);
%         do_lambda_optimization=false;
%         trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
%         if ~any(ismember({'wttf','wtts'}, ...
%                 who('-file',trf_config.model_metric_path)))||overwrite_welch
%             % load preprocessed data/stimuli
%             preprocessed_eeg=load(trf_config.preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
%             preprocessed_eeg=preprocessed_eeg.preprocessed_eeg;
%             cond=preprocessed_eeg.cond;
%             stats_cross_cv=load(trf_config.model_metric_path,"stats_cross_cv");
%             stats_cross_cv=stats_cross_cv.stats_cross_cv;
%             fprintf('running Welch ttest...\n');
%             [wttf,wtts]=welchttest_wrapper(stats_cross_cv,cond,avg_cross_trials_rcross);
%             fprintf('saving Welch ttest result...\n')
%             save(trf_config.model_metric_path,"wttf","wtts","-append")
%         end
%     end
% 
%     fprintf('saving Welch t-test results for subj %d\n',subj);
% end
% 
% %% plot topos of welch t-statistic
% %NOTE: not sure if git will copy this file in which case relative paths
% load(loc_file)
% subjs=[2:7,9:22];
% % subjs=2;
% plotx='tstat';
% 
% switch plotx
%     case 'h'
%         colorlabel='hypothesis test result';
%         clims_=[0,1];
%     case 'p'
%         colorlabel='p-value';
%         clims_='auto';
%     case 'tstat'
%         colorlabel='t-statistic';
%         clims_=[-2 2];
%     otherwise
%         error('not a name.')
% end
% for subj=subjs
%     % load welch ttest results
%     preprocess_config=config_preprocess(subj);
%     do_lambda_optimization=false;
%     trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
%     load(trf_config.model_metric_path,"wttf","wtts")
% 
%     figure
%     if strcmp(plotx,'tstat')
%         topoplot(wttf.stats.(plotx),chanlocs)
%     else
%         topoplot(wttf.(plotx),chanlocs)
%     end
%     h=colorbar;
%     clim(clims_)
%     ylabel(h,colorlabel);
%     title(sprintf('Welch test train on fast subj %d',subj));
% 
%     figure
%     if strcmp(plotx,'tstat')
%         topoplot(wtts.stats.(plotx),chanlocs)
%     else
%         topoplot(wtts.(plotx),chanlocs)
%     end
%     h=colorbar;
%     ylabel(h,colorlabel);
%     clim(clims_)
%     title(sprintf('Welch test train on slow subj %d',subj));
% 
% end

%%


%% GLMM analysis (old)
% 
% subjs=[2:7,9:22];
% 
% disp('GLMM analysis start...')
% tbl=setup_glmm_data(subjs);
% % use mismatched conditions as baseline
% tbl.Match=reordercats(tbl.Match,["false","true"]);
% % fit GLMM
% formula='Rval ~ TrainCond*Match+ (1|Subject) + (1|Subject:Electrode) + (1|DataCond)';
% glme=fitglme(tbl,formula);
% disp(glme)

%% helpers
function cluster_nulldist=get_cluster_nulldist(T_perm,t_thresh,adj)
% T_perm: permutations x conditions x electrodes
% note: when testing on T_perm, need to iterate over each condition AND
% permutation
% will also need to remember to just keep the max cluster for each
% permutation
[n_perm,n_cond,~]=size(T_perm);
cluster_nulldist=nan(n_perm,n_cond);
for nn=1:n_perm
    fprintf('clustering %d/%d...\n',nn,n_perm)
    for cc=1:n_cond
       [~,cluster_masses]=clusterize(squeeze(T_perm(nn,cc,:)),t_thresh,adj); 
       % note: seems like cluster_masses should be a regular array... might
       % need to double check that though
       cluster_nulldist(nn,cc)=max(cluster_masses);
    end
end

end
function [clusters, cluster_masses]=clusterize(t_vals,t_thresh,adj)
    % t_vals should be a col vector of size [n_electrodes X 1] 
    tmask=t_vals>t_thresh;
    %NOTE: if slow... can consider preallocating cell arrays for outputs
    %with size n_electrodes (since there can't be more clusters than
    %electrodes), then shrink the array at the end...?
    clusters={};
    if any(tmask(:))
        % cluster_masses={};
        n_electrodes=length(adj);
        visited=false(n_electrodes,1);
        % exclude below-threshold electrodes from search
        visited(~tmask)=true;
        % define clusters
        while ~(all(visited))
            % start at first above-threshold electrode not yet visited
            ee=find(~visited,1,'first');
            neighbors=find(adj(ee,:));
            % keep only above-threshold neighbors
            neighbors=neighbors(ismember(neighbors,find(tmask)))';
            % check if current electrode (or it's neighbors) contained in an 
            % existing cluster - 
            % if so: add it and it's above-threshold neighbors to that cluster,
            % if not: make a new cluster    
            ee_clust=cellfun(@(x) any(ismember(neighbors,x)), clusters);
            if isempty(ee_clust)||~any(ee_clust)
                % not contained in an existing cluster, initiate a new one
                clusters{end+1}=neighbors;
            else
                % check that ee is only contained within a single cluster
                if sum(ee_clust)>1
                    % if current neighborhood overlaps with multiple existing
                    % clusters, combine them 
                    ee_clust_idx=find(ee_clust);
                    for rr=1:(sum(ee_clust)-1)
                        clusters{ee_clust_idx(1)}=union(clusters{ee_clust_idx(1)},clusters{ee_clust_idx(rr+1)});
                    end
                    % remove excess clusters
                    clusters(ee_clust_idx(2:end))=[];
                    
                end
                ee_clust_idx=find(ee_clust,1,'first');
                new_neighbors=setdiff(neighbors,clusters{ee_clust_idx});
                clusters{ee_clust_idx}=[clusters{ee_clust_idx}; new_neighbors];
            end
    
            % update visitation list
            %todo: also mark rest of in-cluster as visited
            % note: actually, we might want to leave them marked as unvisited,
            % that way we can add their above-threshold neighbors to a cluster
            visited(ee)=true;
        end
        % add up cluster masses
        cluster_masses=cellfun(@(x) sum(t_vals(x)), clusters);
    else
        % no values above threshold-> cluster mass is zero
        cluster_masses=0;
    end

    % check that no clusters have repeated elements
    double_check_=false;
    if double_check_
        % note this is highly inefficient since it'll compute intersection
        % twice for each pair... but figuring out how to fix that felt like
        % a waste of time since the main thing we want to avoid is having
        % overlapping clusters in the first place
        disp(['double checking for fluke clusters... set ' ...
            'to false if convinced this is unnecessary...'])
        clusters_disjoint=check_disjoint(clusters);

    end

end
function [clusters_disjoint, cluster_intersections]=check_disjoint(clusters)
% clusters_disjoint=check_disjoint(clusters)
% clusters_disjoint: bool, clusters: cell arrays of vectors with hopefully
% disjoint integers
    n_clusters=length(clusters);
    cluster_intersections=cell(n_clusters,n_clusters);
    [cx, cy]=ndgrid(1:n_clusters,1:n_clusters);
    cx=cx(:);cy=cy(:);
    for cc=1:length(cx)
        cluster_intersections{cx(cc),cy(cc)}=intersect(clusters{cx(cc)},clusters{cy(cc)});
    end
    ok_clusters=cellfun(@isempty,cluster_intersections);
    % diag comparisons will always be non-empty since each cluster will
    % be its own intersection but we dont care about those
    ok_clusters=ok_clusters+logical(eye(n_clusters));
    if all(ok_clusters(:))
        clusters_disjoint=true;
    else
        clusters_disjoint=false;
        disp('clusters overlap... pls fix...')
    end
end

function adj=get_adjacency_mat(chanlocs)
    coords=[[chanlocs.X]' [chanlocs.Y]' [chanlocs.Z]'];
    D=squareform(pdist(coords));
    dist_thresh=0.40; %TODO: figure out 'correct' value for this
    % it is useful later to consider each electrode a neighbor of itself
    adj=D<dist_thresh;
    % adj=D<dist_thresh&D>0;
    figure, imagesc(adj), axis square; colorbar 
    title(sprintf('electrode ajacancy matrix - dist thresh: %0.2f',dist_thresh))
end

function pairs=get_off_diag_pairs(n_cond)
    [ptrain,ptest]=ndgrid(1:n_cond,1:n_cond);
    pairs=[ptrain(:),ptest(:)];
    % remove diagonals
    pairs(pairs(:,1)==pairs(:,2),:)=[];
end
function [R_within,R_cross]=split_all_subj_Rs(all_subj_Rcs)
% assumes all_subj_Rcs has size [subjs,cond,cond,electrodes]
% returns R_within as [subjs,cond,electrodes]
    [n_subjs,n_cond,~,n_electrodes]=size(all_subj_Rcs);
    % m_w=logical(repmat(eye(n_cond),n_subj,1,1,n_electrodes));
    % R_within=all_subj_Rcs()
    % note: I'm pretty sure there's a way to do this without having to use
    % loops but it's breaking my brain so just gonna say fuckit

    %permute first so looping is more efficient across electrodes:
    all_subj_Rcs=permute(all_subj_Rcs,[4,1,2,3]);
    %preallocate outputs
    R_within=nan(n_electrodes,n_subjs,n_cond);
    R_cross=nan(n_electrodes,n_subjs,n_cond,n_cond-1);
    for ee=1:n_electrodes
        for ss=1:n_subjs
            R_=squeeze(all_subj_Rcs(ee,ss,:,:));
            R_within(ee,ss,:)=diag(R_);
            R_cross(ee,ss,:,:)=reshape(R_(~logical(eye(n_cond))),n_cond,n_cond-1);
            
            % R_cross(ee,ss,:,:)=
        end
    end
    % permute back so subjs is first dim and electrodes last since not
    % really looping through electrodes later
    R_within=permute(R_within,[2 3 1]);
    R_cross=permute(R_cross,[2 3 4 1]);
end
function Rs=compile_rvals(stats_cross_cv,cond,avg_cross_trials)
    n_electrodes=size(stats_cross_cv.r,3);
    n_cond=numel(unique(cond));
    % Rs=cell(n_cond);
    Rs=nan(n_cond,n_cond,n_electrodes);
    % fast_trials=find(cond==1);
    % og_trials=find(cond==2);
    % slow_trials=find(cond==3);
    % flag subjects with missing trials for errors
    cond_ids={find(cond==1),find(cond==2),find(cond==3)};

    % if ~((numel(fast_trials)==numel(og_trials))&&(numel(og_trials)==numel(slow_trials)))
    %     disp('subject has uneven number of trials per condition, handle with care....')
    % end
    if numel(unique((cellfun(@numel, cond_ids))))>1
        disp('subject has uneven number of trials per condition, handle with care.')
    end

    % % extract within-condition predictions
    % R_ff_=stats_cross_cv.r(fast_trials,fast_trials,:);
    % R_oo_=stats_cross_cv.r(og_trials,og_trials,:);
    % R_ss_=stats_cross_cv.r(slow_trials,slow_trials,:);
    % 
    % % remove off-diagonals (nans)
    % 
    % m_ff=logical(repmat(eye(numel(fast_trials)),1,1,n_electrodes));
    % m_oo=logical(repmat(eye(numel(og_trials)),1,1,n_electrodes));
    % m_ss=logical(repmat(eye(numel(slow_trials)),1,1,n_electrodes));
    % 
    % R_ff=R_ff_(m_ff);
    % R_oo=R_oo_(m_oo);
    % R_ss=R_ss_(m_ss);
    %note: I think this square indexing will cause a problem when there is
    %missing trials, so check that nan values at least line up the way we
    %expect first - although presumably the corresponding R_xx_ matrix will
    %be short one row in that case and logical indexing with eye will
    %fail...
    % in case it does not, still check that only nans will be removed:
    % if any(isnan(R_ff_(m_ff)))||any(isnan(R_ff_(m_ss)))||any(isnan(R_oo_(m_oo)))
    %     error('some nans remain')
    % end
    % if ~(all(isnan(R_ff_(~m_ff))))||~(all(isnan(R_ss_(~m_ss))))||all(isnan(R_oo_(m_oo)))
    %     error('some non-nans are missing')
    % end
    % R_ff=reshape(R_ff,[],n_electrodes);
    % R_oo=reshape(R_oo,[],n_electrodes);
    % R_ss=reshape(R_ss,[],n_electrodes);
    % % average the values
    % R_ff=mean(R_ff,1);
    % R_oo=mean(R_oo,1);
    % R_ss=mean(R_ss,1);
    function R_within=get_within(idx)
        R_=stats_cross_cv.r(idx,idx,:);
        m=logical(repmat(eye(numel(idx)),1,1,n_electrodes));
    
        vals=R_(m);
        if any(isnan(vals))
            errors('some nans remain')
        end
        if ~all(isnan(R_(~m)))
            error('some non-nans where nans should be.')
        end
        R_within=mean(reshape(vals,[],n_electrodes),1);
        clear R_
    end
%note: can also replace this loop easily
    % R_ff=get_within(cond_ids{1});
    % R_oo=get_within(cond_ids{2});
    % R_ss=get_within(cond_ids{3});

    for ww=1:n_cond
        % Rs{ww,ww}=get_within(cond_ids{ww});
        Rs(ww,ww,:)=get_within(cond_ids{ww});
    end

    % % load cross-trial rs
    % R_fs=stats_cross_cv.r(fast_trials,slow_trials,:);
    % R_fo=stats_cross_cv.r(fast_trials,og_trials,:);
    % 
    % R_sf=stats_cross_cv.r(slow_trials,fast_trials,:);
    % R_so=stats_cross_cv.r(slow_trials,og_trials,:);
    % 
    % R_of=stats_cross_cv.r(og_trials,fast_trials,:);
    % R_os=stats_cross_cv.r(og_trials,slow_trials,:);
    % 
    % if avg_cross_trials
    %     % average across all training folds
    %     R_fs=mean(R_fs,1);
    %     R_fo=mean(R_fo,1);
    % 
    %     R_sf=mean(R_sf,1);
    %     R_so=mean(R_so,1);
    % 
    %     R_of=mean(R_of,1);
    %     R_os=mean(R_os,1);
    % 
    %     % avg across testing folds
    %     R_fs=mean(R_fs,2);
    %     R_fo=mean(R_fo,2);
    % 
    %     R_sf=mean(R_sf,2);
    %     R_so=mean(R_so,2);
    % 
    %     R_of=mean(R_of,2);
    %     R_os=mean(R_os,2);
    % 
    % end
    % R={R_ff,R_fo,R_fs;...
    % R_of, R_oo, R_os;...
    % R_sf,R_so,R_ss};
    % % R=R';
    
    % extract cross-condition scores

    % (tain indx, test indx)
    % pairs={...
    %     {1,3},{1,2},... %fs,fo
    %     {2,1},{2,3},... %of,os
    %     {3,1},{3,2},... %sf,so
    %     };
    % names={'R_fs','R_fo','R_of','R_os','R_sf','R_so'};
    % [ptrain,ptest]=ndgrid(1:n_cond,1:n_cond);
    % pairs=[ptrain(:),ptest(:)];
    % % remove diagonals
    % pairs(pairs(:,1)==pairs(:,2),:)=[];
    pairs=get_off_diag_pairs(n_cond);
    for kk=1:size(pairs,1)
        train_idx=cond_ids{pairs(kk,1)};
        test_idx=cond_ids{pairs(kk,2)};
        % Rs{pairs(kk,1),pairs(kk,2)}=stats_cross_cv.r(train_idx,test_idx,:);
        R_cross_=stats_cross_cv.r(train_idx,test_idx,:);
        if avg_cross_trials
            % Rs{pairs(kk,1),pairs(kk,2)}=mean(Rs{pairs(kk,1),pairs(kk,2)},1); %avg over train folds
            % Rs{pairs(kk,1),pairs(kk,2)}=mean(Rs{pairs(kk,1),pairs(kk,2)},2); %avg over test folds
            R_cross_=mean(R_cross_,1); %avg over train folds
            R_cross_=mean(R_cross_,2); % avg over test folds
            Rs(pairs(kk,1),pairs(kk,2),:)=permute(squeeze(R_cross_),[2,1]);
        end
        % assignin('base',names{kk},R);
    end
    % Rs={R_ff,R_fo,R_fs,...
    %     R_of, R_oo, R_os,...
    %     R_sf,R_so,R_ss};
end

function [wttf,wtts]=welchttest_wrapper(stats_cross_cv,cond,avg_cross_trials)
% [wttf,wtts]=welchttest_wrapper(stats_cross_cv,cond,avg_cross_trials)
    n_electrodes=size(stats_cross_cv.r,3);
    [R_ff,~,R_fs,...
    ~, ~, ~,...
    R_sf,~,R_ss]=compile_rvals(stats_cross_cv,cond,avg_cross_trials);
    % compute t-test for train fast -> test slow vs train fast -> test fast
    [wttf.h,wttf.p,wttf.ci,wttf.stats]=ttest2(R_ff, ...
        reshape(R_fs,[],n_electrodes),"Vartype","unequal");
    % compute t-test for train slow -> test fast vs train slow -> test slow
    [wtts.h,wtts.p,wtts.ci,wtts.stats]=ttest2(R_ss, ...
        reshape(R_sf,[],n_electrodes),"Vartype","unequal");
end

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
    % for subj=subjs
    %     tmp_pcnfg=config_preprocess(subj);
    %     tmp_tcnfg=config_trf(subj,false,tmp_pcnfg);
    %     tmp_sc=load(tmp_tcnfg.model_metric_path,"stats_cross"); % should only have 3x3 shaped vars
    %     tmp_sc=tmp_sc.stats_cross;
    %     for ii=1:3 % data condition
    %         for jj=1:3 % training condition
    %             R=tmp_sc(ii,jj).r;
    %             [n_trials,~,n_electrodes]=size(R);
    %             for tt=1:n_trials
    %                 for ee=1:n_electrodes
    %                     Rval(row)=R(tt,1,ee);
    %                     Subject(row)=categorical(subj); % does it have to be a string?
    %                     DataCond(row)=categorical(ii);
    %                     TrainCond(row)=categorical(jj);
    %                     Match(row)=categorical(DataCond(row)==TrainCond(row));
    %                     Electrode(row)=ee;
    %                     row=row+1;
    %                 end
    %             end
    %         end
    %     end
    % end
    for subj=subjs
        tmp_pcnfg=config_preprocess(subj);
        tmp_tcnfg=config_trf(subj,false,tmp_pcnfg);
        tmp_sc=load(tmp_tcnfg.model_metric_path,"stats_cross_cv"); 
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


