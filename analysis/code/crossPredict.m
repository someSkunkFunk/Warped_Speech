% crossPredict.m
%
% PURPOSE:
%   Test whether EEG/TRF models trained on one speech condition generalize
%   to neural responses recorded during a different speech condition. E.g. 
%   If a model trained on "fast" speech can predict EEG to "slow" speech as
%   well as a model trained on "slow" speech itself, that suggests a shared
%   underlying neural representation across conditions.

% INTENDED GOAL (vs. current implementation):
%   The long-term goal is to asses cross-condition generalization as
%   evidence for condition-invariant auditory-cortical tracking. The
%   current script, however, tests a simpler proxy question: is
%   within-condition prediction accuracy (R_within) significantly greater
%   than cross-condition prediction accuracy (R_cross)? A significant
%   difference indicates that models are at least partially
%   condition-specific, i.e. do NOT fully generalize.

% METHOD:
%   Implemenets the Maris & Oostenveld (2007) cluster-based nonparametric
%   permutation test across subjects. For each subject:
%   1. A TRF model is trained on held-out folds of one conditioin (LOOCV)
%   and its correlation (r) with both within- and cross-condition EEG is
%   recorded yielding a [trials x trials x electrodes] matrix of r-values. 
%   2. These are collapsed into a [conditions x conditions x electrodes]
%   matrix of mean r-values per subject (Rcs).


clear, clc
%% --- CONFIGURATION ---
plots_config=[]; %todo... defaults + use wrapper function
plots_config.show_ind_subj=false;
% subjs=[2:7,9:22];
% subjs=[2:7,9:22];

subjs=[2:7,9:12];
script_config.show_tuning_curves=false;
script_config.compute_cross_stats=true;
script_config.overwrite_stats_cross=false;
% trim end of trials for og/slow to match number of samples in fast trials
script_config.trim_stimuli=true;
script_config.get_all_subj_Rcs=true;
script_config.avg_cross_trials_rcross=true;
% note: not sure how to interpret non-averaged results so non-averaged
% cross-trial reslts are not currently handled downstreamm
script_config.overwrite_Rcs=false;

%% --- COMPUTE CROSS-PREDICTION STATISTICS (per subject, LOOCV) ---
% For each subject, trains a TRF model using LOOCV. Within each condition,
% then evaluates prediction accuracy on:
%   (1) the held-out trial from the same conditions -> fills diagonal of
%   stats_cross_cv.r
%   (2) all trials from every other condition -> fills off-diagonal entries
% results is saved as stats_cross_cv: a [trials x trials x electrodes]
% struct of r/err.


if script_config.compute_cross_stats
    fair=true;
    rng(1); % seed for reproducibility
    for subj=subjs
        fprintf('computing stats_cross for subj %d...\n',subj)
        % load saved model
        trf_analysis_params;
        clear do_nulltest
        trf_=load_checkpoint(trf_config);
        if ~isfield(trf_,'stats_cross_cv')||script_config.overwrite_stats_cross
            pp_=load_checkpoint(preprocess_config);
            preprocessed_eeg=pp_.preprocessed_eeg;
            stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
            [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
                trf_config);
            cond_labels=preprocessed_eeg.cond;
            resp=preprocessed_eeg.resp';
                
            if script_config.trim_stimuli
            % trim stimuli so all conditions have equal number of samples
                min_ns=min(cellfun('size',stim,1));
                stim=cellfun(@(x) x(1:min_ns),stim,'UniformOutput',false);
                resp=cellfun(@(x) x(1:min_ns,:),resp,'UniformOutput',false);
            end
        
            if fair
                % shuffle trial order so LOOCV folds are not
                % condition-contiguous
                best_lam=train_params.best_lam;
                shuff=randperm(numel(cond_labels));
                cond_labels=cond_labels(shuff);
                resp=resp(shuff);
                stim=stim(shuff);

                n_trials=numel(cond_labels);
                n_electrodes=size(resp{1},2);
                %initialize stats_cross_cv.r(i,j,e):
                % r-value when model was trained on the fold that held out
                % trial i, evaluated on trial j, at electrode e.
                % Diagonal (i==j): within-condition held-out prediction.
                % off-diagonal: cross-condition prediction.

                stats_cross_cv=struct( ...
                    'r',nan(n_trials,n_trials,n_electrodes), ...
                    'err',nan(n_trials,n_trials,n_electrodes));
        
                for cc=1:3
                    fprintf('%d of 3 conditions...\n',cc);
                    n_folds=sum(cond_labels==cc);
                    cc_trials=find(cond_labels==cc);
                    cross_trials=find(cond_labels~=cc);
                    for k=1:n_folds
                        fprintf('%d of %d folds...\n',k,n_folds)
                        test_trial=cc_trials(k);
                        train_trials=cc_trials(cc_trials~=test_trial);
                        % train model on all same-condition trials except
                        % held out one
                        temp_model=mTRFtrain(stim(train_trials),resp(train_trials), ...
                            preprocess_config.fs,1,trf_config.tmin_ms, ...
                            trf_config.tmax_ms,best_lam);
                        % evaluate on the held-out trial AND all
                        % cross-condition trials
                        [~,temp_stats]=mTRFpredict([stim(test_trial);stim(cross_trials)] ...
                            ,[resp(test_trial);resp(cross_trials)],temp_model);
                        % store within-condition (diagonal) result
                        stats_cross_cv.r(test_trial,test_trial,:)=temp_stats.r(1,:);
                        stats_cross_cv.err(test_trial,test_trial,:,:)=temp_stats.err(1,:);
                        % store cross-condition (off-diagonal) results
                        stats_cross_cv.r(test_trial,cross_trials,:)=temp_stats.r(2:end,:);
                        stats_cross_cv.err(test_trial,cross_trials,:)=temp_stats.err(2:end,:);
                    end 
                end
                % invert the shuffle so saved indices correspond to the
                % original trials order stored in preprocess_config
                stats_cross_cv.r(shuff,shuff,:)=stats_cross_cv.r;
                stats_cross_cv.err(shuff,shuff,:)=stats_cross_cv.err;
                %save to file
                save_checkpoint(stats_cross_cv,trf_config,script_config.overwrite_stats_cross)
            else
                error('DONT DO THIS')
                % % unfair (non-LOOCV) comparison, kept for reference
                % model_data=load(trf_config.trf_model_path,"model");
                % stats_data=load(trf_config.model_metric_path,"stats_obs");
                % stats_obs=stats_data.stats_obs(2,:); clear stats_data
                % model=model_data.model(2,:); clear model_data
                % conditions_=1:3;
                % stats_cross=cell2struct(cell(2,1),fieldnames(stats_obs));
                % 
                % % train_condition, predict_condition, electrodes
                % for cc=conditions_
                %     % copy paste same-condition r values from nulldistribution file
                %     % r_cross(cc,cc,:)=stats_obs.r(cc,:);
                %     stats_cross(cc,cc)=stats_obs(cc); % assumes 1D struct array...
                %     for icc=conditions_(conditions_~=cc)
                %         [~,stats_cross(cc,icc)]=mTRFpredict(stim(cond==cc),resp(cond==cc),model(icc));
                %     end
                % end
                % clear cc 
                % fprintf('saving cross-prediction results for subj %d\n',subj)
                % save(trf_config.model_metric_path,"stats_cross","-append")
                % fprintf('result saved.\n')
            end
        else
            disp('existing stats_cross found')
        end
        clear trf_
    end
end
%% --- COMPILE PER-SUBJECT R-VALUE MATRICES (Rcs) ---
% Collapses stats_cross_cv (trial x trial x electrode) into Rcs: 
% a [conditions x conditions x electrodes] matrix of mean r-values per
% subject. 
% Diagonal entrials = within-condition r; off-diagonal = cross-condition r. 
% All subjects are stacked into all_subj_Rcs [subjects x
% cond x cond x electrodes].
if script_config.get_all_subj_Rcs 
    % note: could improve by pre-allocating?
    n_electrodes=128; % how to avoid hardcoding? does it even matter?
    n_cond=3;
    n_subjs=numel(subjs);
    all_subj_Rcs=nan(n_subjs,n_cond,n_cond,n_electrodes);
    do_lambda_optimization=false;
    for ss=1:numel(subjs)
        subj=subjs(ss);
        % load saved model
        fprintf('fetching subj %d Rcs...\n',subj);
        % load saved model
        trf_analysis_params;
        clear do_nulltest
        trf_=load_checkpoint(trf_config);
        if ~isfield(trf_,'Rcs')||script_config.overwrite_Rcs
            fprintf('compiling subj %d Rcs...\n',subj);
            % load preprocessed data/stimuli
            pp_=load_checkpoint(preprocess_config);
            exp_cond_=pp_.preprocessed_eeg.cond;
            clear preprocessed_eeg pp_
            stats_cross_cv=trf_.stats_cross_cv;
            % [R_ff,R_fo,R_fs,...
            % R_of, R_oo, R_os,...
            % R_sf,R_so,R_ss]
            Rcs=compile_rvals(stats_cross_cv,exp_cond_,script_config.avg_cross_trials_rcross);
            fprintf('saving Rcs for subj %d...\n',subj);
            save_checkpoint(Rcs,trf_config,script_config.overwrite_Rcs)
        else
            fprintf('loading from saved file...\n')
            S_=load_checkpoint(trf_config);
            Rcs=S_.Rcs;
            clear S_
        end
        clear trf_ exp_cond_
        all_subj_Rcs(ss,:,:,:)=Rcs; 
    end
end
%% --- ORGANIZE R-VALUES AND DEFINE CROSS-CONDITION PAIRS ---
% separates all_subj_Rcs into:
%   R_within [subj cond x electrodes]: diagonal entries
%   R_cross [subj x train_cond x test_cond x electrodes]: off-diagonal entries
% also enumerates all ordered (train, test) cross-condition pairs.

cond_labels=trf_config.conditions;
[R_within,R_cross]=split_all_subj_Rs(all_subj_Rcs);
% R_within: [subj x train_cond x electrodes]
% R_cross:  [subj x train_cond x test_cond x electrodes]

off_diag_pairs=get_off_diag_pairs(n_cond); % [n_cross x 2]: [train_cond, test_cond]
n_cross=length(off_diag_pairs);
% adjust train index to account for the missing test condition dimension in
% R_cross
cross_train_idx=off_diag_pairs(:,1);
cross_train_idx(off_diag_pairs(:,1)>off_diag_pairs(:,2))= ...
cross_train_idx(off_diag_pairs(:,1)>off_diag_pairs(:,2))-1;

% build readable labels for each cross-condition pair (e.g.
% "fast,slow,...")
cross_ids=cell(size(off_diag_pairs,1),1);
for pp=1:n_cross
    cross_ids{pp}=sprintf('%s,%s',cond_labels{off_diag_pairs(pp,1)},cond_labels{off_diag_pairs(pp,2)});
end


%% --- load chanlocs ---
global boxdir_mine
loc_file=sprintf("%s/data/128chanlocs.mat",boxdir_mine);
load(loc_file);

%%  --- OOSTENVELD PERMUTATION TEST ---

%% --- COMPUTE OBSERVED CONTRAST (D_obs) AND PERMUTED CONTRAST (D_perm) ---
% D is the contrast between within- and cross-condition prediction
% accuracy, either as a raw difference or percent change (set via
% D_config.metric). D_config.comparison controls whether cross-condition
% predictions from all training 
% conditions are averaged together ('lumped')
% or kept separate ('separate').
% The permutation null distribution is built by randomly shuffling
% condition labels (the train-condition dimension of all_subj_Rcs) within
% each subject, independently across subjects,n_perm times.

% TODO: I think this is where we have to account for the within-subject
% design.

D_config=[]; 
D_config.metric='percent_change'; % 'percent_change' or 'raw_diff'
D_config.comparison='separate'; % 'separate' or 'lumped'
disp(D_config)
[D_obs,D_config]=get_D(R_within,R_cross,D_config);
n_comp=size(D_obs,1);

n_perm=6000;
% generate permutation indices: one permutation of condition labels per
% subject per draw -- applied across all train conditions
[~,perm_idx]=sort(rand(n_perm,n_subjs,n_cond), 3);

% build index grids to vectorize permutation indexing (avoids 5D nested loops)
[perm_grid,subj_grid,train_grid,test_grid,elec_grid]=ndgrid(1:n_perm,1:n_subjs, ...
    1:n_cond,1:n_cond,1:n_electrodes);
% pray for no memory issues

% Replicate perm_idx across electrode and test condition dimensions 
% for vectorized sub2ind call
perm_train_grid=perm_idx(:,:,:,ones(1,n_cond),ones(1,n_electrodes));

lin_perm_idx=sub2ind(size(all_subj_Rcs),subj_grid(:),perm_train_grid(:), ...
    test_grid(:),elec_grid(:));
% apply permutations to all_subj_Rcs to get shuffled r-value arrays
perm_Rcs=reshape(all_subj_Rcs(lin_perm_idx),n_perm,n_subjs,n_cond, ...
    n_cond,n_electrodes);


% compute D for each permutation
D_perm=nan(n_comp,n_perm,n_subjs,n_cond,n_electrodes);
for pp=1:n_perm
    D_perm(:,pp,:,:,:)=get_D(squeeze(perm_Rcs(pp,:,1,:,:)) ...
        ,squeeze(perm_Rcs(pp,:,2:end,:,:)),D_config);
end


%% --- Compute group-level statistics ---
% one-sample t-statistic across subjects for both observed and permuted
% contrasts. t_thresh is the two-tailed critical value used to threshold
% electrode maps before clustering (following Maris & oostenveld 2007).

crit_p=0.05;
t_thresh=abs(tinv(crit_p,n_subjs-1));

T_perm=nan(n_comp,n_perm,n_cond,n_electrodes);
T_obs=nan(n_comp,n_cond,n_electrodes);

for dd=1:size(D_obs,1)
    % compute across-subjects t-statistics %TODO: SHOULD BE
    % WITHIN-SUBJECTS
    D_perm_=D_perm(dd,:,:,:,:);
    T_perm(dd,:,:,:)=squeeze(mean(D_perm_,3)./(std(D_perm_,0,3)/sqrt(n_subjs)));
    clear D_perm_
    D_obs_=D_obs(dd,:,:,:);
    T_obs(dd,:,:)=squeeze(mean(D_obs_,2)./(std(D_obs_,0,2)/sqrt(n_subjs)));
    clear D_obs_
end
%% --- Visualize t-statistic and contrast topographies

% Topoplots of across-subjectt-statistics (one per comparison)
for dd=1:n_comp
    for cc=1:n_cond
        figure
        topoplot(T_obs(dd,cc,:),chanlocs);
        colorbar
        title(sprintf('t-statistic for %s',cond_labels{cc}))
    end
end

% Optional: per-subject D topoplot
if plots_config.show_ind_subj
    for dd=1:n_comp
        for cc=1:n_cond
            for ss=1:n_subjs
                figure
                topoplot(D_obs(dd,ss,cc,:),chanlocs);
                colorbar
                title(sprintf('subj %d R_{within}-mean(R_{cross}) - %s',...
                    subjs(ss),cond_labels{cc}))
            end
        end
    end
end

% --- scatter plots D_obs & T_obs ---
%TODO: configure these to work in lumped vs not comparisons case... 
for dd=1:n_comp
    pretty_scat(squeeze(T_obs(dd,:,:)),1);
    % there must be a way to do the same plot in one line - also we want to add
    % lines i think
    % figure
    % scatter(repmat(cc,n_electrodes,1),squeeze(T_obs(cc,:)));
    title(sprintf('across-subjects t-statistic - all electrodes'))
    xlabel('condition')
    switch D_config.metric
        case 'raw_diff'
            ylabel('tstat(R_{within}-mean(R_{cross}))')
        case 'percent_change'
            ylabel('tstat(% \Delta(R_{within}, mean(R_{cross}))) ')
    end
    xticks(1:3)
    xticklabels(cond_labels)
    hold off
end
% SFN PLOT HERE (summary of t-stats for all cross-condition pairs)
figure('Name','SFN-plot','Color','white')
pretty_scat(reshape(T_obs,n_cross,n_electrodes),1)
title('Scalp-EEG Crossed-Conditions Prediction Accuracy')
xlabel('i,j (train,test)')
ylabel('t-stat(C{i,j})')
xticks(1:n_cross)
xticklabels(cross_ids)
set(gca,'FontSize',28)

% Scatterplot R_within and R_Cross (group level)
figure ('Name','R_within - group level')
pretty_scat(R_within,2)

title('all subj & all electrodes')
xlabel('condition')
ylabel('R_{within}')
xticks(1:n_cond)
xticklabels(cond_labels)
hold off

% R_cross: loop version kept for cross-checking vectorized
% TODO: check indexing... train/test have been swapped (actually I don't think it matters here
figure
for pp=1:n_cross
    R_=squeeze(R_cross(:,cross_train_idx(pp),off_diag_pairs(pp,2),:));
    R_=R_(:);
    scatter(repmat(pp,n_electrodes*n_subjs,1),R_);
    hold on
    title('loop result for reference')
end
% R_cross: vectorized 
pretty_scat(reshape(R_cross,n_subjs,n_cross,n_electrodes),2)
title('all subjs & all electrodes')
xlabel('train->test')
ylabel('R_{cross}')
xticks(1:n_cross)
xticklabels(cross_ids)
hold off
% --- OPTIONAL subject-level R_scatterplots ---
% rwithin

if plots_config.show_ind_subj
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
        xticklabels(cond_labels)
        hold off
    end
    
    %rcross
    for ss=1:n_subjs
        for pp=1:n_cross
            cross_ids{pp}=sprintf('%s,%s',cond_labels{off_diag_pairs(pp,1)},cond_labels{off_diag_pairs(pp,2)});
        end
        figure
        for pp=1:n_cross
            R_=squeeze(R_cross(:,off_diag_pairs(pp,1),off_diag_pairs(pp,2),:));
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
end

%% --- define electrode neighborhoods ---
% Builds an adjacency matrix over electrodes based on 3D euclidian
% distance. used by cluster algorithm to define spatially connected
% electrode groups.
adj_config=[];
% note: set true to generate one figure per electrode 
% (will generate 129 figures!)
adj_config.vis_neighbors=false; 
adj=get_adjacency_mat(chanlocs,adj_config);


%% --- CLUSTER-BASED PERMUTATION TEST ---
% for each comparison and condition:
%   1. identify clusters of spatially adjacent electrodes where T_obs >
%       t_thresh.
%   2. Compute the mass (sum of t-values) of each cluster.
%   3. Compare the observed cluster masses against the null distribution of
%   max cluster mass obtained from the permuted T_perm maps.
%   4. Clusters whose mass exceed the (1 - crip_p) quantile of the null
%   distribution are considered significant.
obs_clusters=cell(n_comp,n_cond,1);obs_masses=cell(n_comp,n_cond,1);
cluster_nulldist=cell(n_comp,1);
for dd=1:n_comp
    fprintf('comp %d of %d\n',dd,n_comp)
    for cc=1:n_cond
        fprintf('getting observed cluster statistics')
        [obs_clusters{dd,cc}, obs_masses{dd,cc}]=clusterize(squeeze(T_obs(dd,cc,:)),t_thresh,adj);
    end

    % do cluster test on T_perm
    disp('generating permutation distributions')
    cluster_nulldist{dd}=get_cluster_nulldist(squeeze(T_perm(dd,:,:,:)),t_thresh,adj);
end

% plot empirical CDFs of null distributions with critical mass threshold
crit_mass=nan(n_cond,1);
for dd=1:n_comp
    for cc=1:n_cond
        nulldist_=cluster_nulldist{dd};
        figure
        [F_null_,Fmass_null_]=ecdf(nulldist_(:,cc));
        %note: need to call twice to plot? that's outrageous
        ecdf(nulldist_(:,cc));
        hold on
        % get critical mass
        crit_mass(cc)=min(Fmass_null_(F_null_>(1-crit_p)));
        plot(repmat(crit_mass(cc),2,1), [0 1], 'r--')
        xlabel('cluster mass')
        ylabel('probability')
        legend('cdf',sprintf('crit mass: %0.1f',crit_mass(cc)))
        title(sprintf('%s null',cond_labels{cc}))
        hold off
        clear Fmass_null_ F_null_
    end
end

%% --- topoplot significant clusters (electrodes belonging to clusteers above crit_mass) ---
for cc=1:n_cond
    
    
    clusts_=obs_clusters{cc};
    masses_=obs_masses{cc};
    sig_clusts_=masses_>crit_mass(cc);

    
    fprintf('for %s, any clusters observed above %0.2f threshold? %d\n', ...
        cond_labels{cc},crit_p,...
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
            title(sprintf('test cond:%s, cluster %d of %d',cond_labels{cc},sc,numel(sig_clusts_)))
            clear clust_ts

        end

    end

    clear clusts_ masses_ sig_clusts_
end




%% helpers
function pretty_scat(D,cond_dim)
% PRETTY_SCAT scatterplot with per-sample connecting lines and boxplots.
%
% pretty_scat(D, cond_dim)
%
% D - data matrix of any dimensionality
% cond_dim - index of the dimension to use at the x-axis (condition/pairs)
%
% Permutes D so that cond_dim is last, then reshapes to [n_smaples x
% n_cond] and plots grey connecting lines, scatter points, and boxplots.
sz = size(D);
n_cond = sz(cond_dim);

% Permute so that condition dimension is last
if cond_dim ~= numel(sz)
    dim_ord = 1:numel(sz);
    dim_ord = [dim_ord(dim_ord~=cond_dim), cond_dim];
    D = permute(D, dim_ord);
    sz = size(D);
end

n_per_cc = prod(sz(1:end-1));

figure('Units','inches','Position',[0 0 8 5]); hold on

% Make everything grey
point_color = [0.5 0.5 0.5];
line_color = [0.7 0.7 0.7];

% Reshape so we can plot across conditions easily
Dmat = reshape(D, [n_per_cc, n_cond]);

% --- 1) Draw connecting lines across each row (each sample)
for ii = 1:n_per_cc
    plot(1:n_cond, Dmat(ii,:), '-', 'Color', line_color, 'LineWidth', 0.5);
end

% --- 2) Scatter points
scatter(repmat(1:n_cond, n_per_cc, 1), Dmat, 15, point_color, 'filled', ...
    'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');

% --- 3) Add boxplots
boxplot(Dmat, 'Colors', [0.2 0.2 0.2], 'Symbol', '', 'Widths', 0.4);

% --- 4) Beautify axes
set(gca, 'XTick', 1:n_cond, 'Box', 'off', 'TickDir', 'out');
xlabel('Condition');
ylabel('Value');
title('Pretty scatter with lines and boxplots');
xlim([0.5 n_cond+0.5]);
end

function [D,config]=get_D(R_within,R_cross,config)
% GET_D Compute contrast between within- and cross-condition predicition
% accuracies
% [D, config] = get_D(R_within, R_cross, config)
% R_within: subjs-by-test_conditions-by-channels
% R_cross: subjs-by-train_conditions-by-test_conditions-by-channels
% config - struct with fields: 
%   .metric - 'raw_diff' or 'percent_change'
%   .comparison - 'lumped' (average over train conditions) or 'separate'
%
% D - contrast array; first dimension indexes train conditions.
% note: set true to generate one figure per electrode (will generate 129 figures!)

defaults=struct( ...
    'metric','raw_diff', ...
    'comparison','lumped');
fldnms=fieldnames(defaults);
for ff=1:numel(fldnms)
    fldnm=fldnms{ff};
    if ~isfield(config,fldnm)||isempty(config.(fldnm))
        config.(fldnm)=defaults.(fldnm);
    end
end

switch lower(config.comparison)
    case 'lumped'
        % don't distinguish between cross-training conditions
        R_cross=mean(R_cross,2);
    case 'separate'
        % can just permute and use first dim for iteration later
    otherwise
        error('config.comparison:%s not valid',config.comparison)
end
% make training condition(s) first dimension
R_cross=permute(R_cross,[2 1 3 4]);
D=nan(size(R_cross));
for dd=1:size(D,1)
    % get R_cross in compatible shape with R_within
    R_cross_=squeeze(R_cross(dd,:,:,:));
    switch lower(config.metric)
        case 'raw_diff'
            D(dd,:,:,:)=R_within-R_cross_;
            % D_obs=R_within-squeeze(mean(R_cross,3));
        case 'percent_change'
            D(dd,:,:,:)=100*(R_within-R_cross_)./R_within;
        otherwise
            error('config.metric: %s not valid',config.metric)
    end
end

end
function cluster_nulldist=get_cluster_nulldist(T_perm,t_thresh,adj)
% GET_CLUSTER_NULLDIST  Build the null distribution of maximum cluster mass.
%
%   cluster_nulldist = get_cluster_nulldist(T_perm, t_thresh, adj)
%
%   For each permutation and condition, finds spatial electrode clusters in the
%   permuted t-map, computes their mass (sum of t-values), and records the
%   maximum. The resulting [n_perm x n_cond] matrix is the null distribution
%   against which observed cluster masses are compared.
%
%   T_perm   - [n_perm x n_cond x n_electrodes]
%   t_thresh - scalar threshold for inclusion in a cluster
%   adj      - [n_electrodes x n_electrodes] logical adjacency matrix(T_perm);
cluster_nulldist=nan(n_perm,n_cond);
for nn=1:n_perm
    fprintf('clustering %d/%d...\n',nn,n_perm)
    for cc=1:n_cond
       [~,cluster_masses]=clusterize(squeeze(T_perm(nn,cc,:)),t_thresh,adj); 
       cluster_nulldist(nn,cc)=max(cluster_masses);
    end
end

end
function [clusters, cluster_masses]=clusterize(t_vals,t_thresh,adj)
% CLUSTERIZE  Group suprathreshold electrodes into spatially connected clusters.
%
%   [clusters, cluster_masses] = clusterize(t_vals, t_thresh, adj)
%
%   Implements a breadth-first-style graph traversal over the electrode
%   adjacency matrix. Starting from each unvisited suprathreshold electrode,
%   collects its above-threshold neighbours into a cluster. If a new
%   electrode's neighbourhood overlaps multiple existing clusters, those
%   clusters are merged. Returns the cluster mass (sum of t-values within
%   each cluster) used as the test statistic in the permutation test.
%
%   t_vals   - [n_electrodes x 1] vector of t-statistics
%   t_thresh - scalar threshold; only electrodes above this are clustered
%   adj      - [n_electrodes x n_electrodes] logical adjacency matrix
%              (self-adjacency on the diagonal is assumed)
%
%   clusters       - cell array; each cell contains electrode indices for one cluster
%   cluster_masses - vector of summed t-values per cluster (0 if none found)
    tmask=t_vals>t_thresh;

    clusters={};
    if any(tmask(:))

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
% CHECK_DISJOINT  Verify that no electrode belongs to more than one cluster.
%
%   [clusters_disjoint, cluster_intersections] = check_disjoint(clusters)
%
%   Computes pairwise intersections of all cluster index vectors. Returns
%   true if all off-diagonal intersections are empty (clusters are disjoint).
%   Used as a debugging check inside clusterize when double_check_ is true.
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

function adj=get_adjacency_mat(chanlocs,config)
% GET_ADJACENCY_MAT  Build a binary electrode adjacency matrix from 3D coordinates.
%
%   adj = get_adjacency_mat(chanlocs, config)
%
%   Two electrodes are considered neighbours if their Euclidean distance is
%   below dist_thresh (default 0.35, in the same units as chanlocs XYZ).
%   The diagonal is included (each electrode is its own neighbour), which
%   simplifies cluster mass calculations.
%
%   chanlocs     - EEGLAB channel location struct with fields X, Y, Z
%   config       - struct with optional fields:
%                    .dist_thresh  - distance threshold (default 0.35)
%                    .vis_neighbors - if true, plot each electrode's neighbourhood
    defaults=struct( ...
        'dist_thresh',0.35, ...
        'vis_neighbors',false);
    fields=fieldnames(defaults);
    %extract defaults
    for ff=1:numel(fields)
        field=fields{ff};
        if ~isfield(config,field)||isempty(config.(field))
            config.(field)=defaults.(field);
        end
    end
    dist_thresh=config.dist_thresh; %TODO: figure out 'correct' value for this
    vis_neighbors=config.vis_neighbors;
    % establish coordinates and get pairwise distances
    coords=[[chanlocs.X]' [chanlocs.Y]' [chanlocs.Z]'];
    D=squareform(pdist(coords));
    
    % it is useful later to consider each electrode a neighbor of itself
    adj=D<dist_thresh;
    % adj=D<dist_thresh&D>0;
    figure, imagesc(adj), axis square; colorbar 
    title(sprintf('electrode ajacancy matrix - dist thresh: %0.2f',dist_thresh))
    if vis_neighbors
        warning('Will generate 129 plots.')
    % check adjacency matrix
        for chn_=1:n_electrodes
            figure
            topoplot([],chanlocs,'electrodes','on','style','blank','plotchans', ...
                find(adj(chn_,:)),'emarker',{'o','b',10,1},'headrad',.5);
            title(sprintf('%s + neighbors, distance threshold: %0.3f', ...
                chanlocs(chn_).labels),dist_thresh)
        end
    end
end


function pairs=get_off_diag_pairs(n_cond)
% GET_OFF_DIAG_PAIRS  Enumerate all ordered off-diagonal index pairs for an n_cond grid.
%
%   pairs = get_off_diag_pairs(n_cond)
%
%   Returns an [n_cross x 2] matrix where each row is [train_cond, test_cond]
%   for every cross-condition combination (diagonal pairs excluded).
%   Sorted by test condition.
    [ptrain,ptest]=ndgrid(1:n_cond,1:n_cond);
    pairs=[ptrain(:),ptest(:)];
    % remove diagonals
    pairs(pairs(:,1)==pairs(:,2),:)=[];
    % sort them by test condition
    [~,Ipairs]=sort(pairs(:,2));
    pairs=pairs(Ipairs,:);
end

function [R_within,R_cross]=split_all_subj_Rs(all_subj_Rcs)
% SPLIT_ALL_SUBJ_RS  Separate diagonal (within) and off-diagonal (cross) r-values.
%
%   [R_within, R_cross] = split_all_subj_Rs(all_subj_Rcs)
%
%   all_subj_Rcs - [n_subjs x n_cond x n_cond x n_electrodes]
%
%   R_within  - [n_subjs x n_cond x n_electrodes]         (diagonal entries)
%   R_cross   - [n_subjs x (n_cond-1) x n_cond x n_electrodes] (off-diagonal)
%
%   The off-diagonal entries are reshaped to [n_cond-1 x n_cond] per subject/electrode,
%   where rows index which training condition is missing from the test set.
    [n_subjs,n_cond,~,n_electrodes]=size(all_subj_Rcs);

    %permute first so looping is more efficient across electrodes:
    all_subj_Rcs=permute(all_subj_Rcs,[4,1,2,3]);
    %preallocate outputs
    R_within=nan(n_electrodes,n_subjs,n_cond);
    R_cross=nan(n_electrodes,n_subjs,n_cond-1,n_cond);
    for ee=1:n_electrodes
        for ss=1:n_subjs
            R_=squeeze(all_subj_Rcs(ee,ss,:,:));
            R_within(ee,ss,:)=diag(R_);
            R_cross(ee,ss,:,:)=reshape(R_(~logical(eye(n_cond))),n_cond-1,n_cond);
            

        end
    end
    % permute back so subjs is first dim and electrodes last since not
    % really looping through electrodes later
    R_within=permute(R_within,[2 3 1]);
    R_cross=permute(R_cross,[2 3 4 1]);
end
function Rs=compile_rvals(stats_cross_cv,cond,avg_cross_trials)
    % COMPILE_RVALS  Collapse the trial-level r-value matrix into a per-condition summary.
%
%   Rs = compile_rvals(stats_cross_cv, cond, avg_cross_trials)
%
%   Takes stats_cross_cv.r [trials x trials x electrodes], where entry (i,j,e)
%   is the r-value when the model (trained leaving trial i out) predicted trial j
%   at electrode e. Collapses this into Rs [n_cond x n_cond x electrodes] by:
%     - Diagonal: averaging r-values of the within-condition LOOCV held-out predictions.
%     - Off-diagonal: averaging r-values across all train/test trial pairings for
%       each cross-condition combination (if avg_cross_trials is true).
%
%   stats_cross_cv  - struct with field .r [trials x trials x electrodes]
%   cond            - [n_trials x 1] condition label vector
%   avg_cross_trials - logical; if true, average over trial pairs (currently assumed true)

    n_electrodes=size(stats_cross_cv.r,3);
    n_cond=numel(unique(cond));
    % preallocate
    Rs=nan(n_cond,n_cond,n_electrodes);
    cond_ids={find(cond==1),find(cond==2),find(cond==3)};


    if numel(unique((cellfun(@numel, cond_ids))))>1
        % flag subjects with missing trials for errors
        disp('subject has uneven number of trials per condition, handle with care.')
        disp('number of trials per condition:')
        disp(cellfun(@numel,cond_ids))
    else
        fprintf('all conditions have %d trials.\n',numel(cond_ids{1}));
    end

    
    function R_within=get_within(idx)
        % GET_WITHIN  Average the LOOCV diagonal r-values for a set of trial indices.
        %   Extracts the [idx x idx] submatrix of stats_cross_cv.r, takes the diagonal
        %   (one held-out prediction per fold), and averages across folds.
        % trials x trials x electrodes
        R_=stats_cross_cv.r(idx,idx,:);
        m=logical(repmat(eye(numel(idx)),1,1,n_electrodes));
    
        vals=R_(m); % unfolds into col vector?
        if any(isnan(vals))
            error('some nans remain')
        end
        if ~all(isnan(R_(~m)))
            error('some non-nans where nans should be.')
        end
        % average across trials of current condition
        R_within=mean(reshape(vals,[],n_electrodes),1);
        clear R_
    end

    for ww=1:n_cond
        Rs(ww,ww,:)=get_within(cond_ids{ww});
    end

    % extract cross-condition scores

    % (tain indx, test indx)
    pairs=get_off_diag_pairs(n_cond);
    for kk=1:size(pairs,1)
        fprintf('train, test: %d,%d\n',pairs(kk,1),pairs(kk,2))
        train_idx=cond_ids{pairs(kk,1)};
        test_idx=cond_ids{pairs(kk,2)};
        disp('checking that train idx conditions are all equal:')
        disp(cond(train_idx))
        disp('checking that quantity of trials makes sense')
        disp(numel(cond(train_idx)))
        disp('check that test idx conditions are all equal:')
        disp(cond(test_idx))
        disp('check that quantity of trials makes sense')
        disp(numel(cond(test_idx)))
        % Rs{pairs(kk,1),pairs(kk,2)}=stats_cross_cv.r(train_idx,test_idx,:);
        R_cross_=stats_cross_cv.r(train_idx,test_idx,:);
        disp('checking that R_cross_ size makes sense')
        disp(size(R_cross_))
        if avg_cross_trials
            % average out across all trials for current cross-condition
            % pairing
            R_cross_=mean(R_cross_,1); %avg over train folds
            R_cross_=mean(R_cross_,2); % avg over test folds
            Rs(pairs(kk,1),pairs(kk,2),:)=R_cross_;
        end
    end
    % Rs={R_ff,R_fo,R_fs,...
    %     R_of, R_oo, R_os,...
    %     R_sf,R_so,R_ss};
end

function [wttf,wtts]=welchttest_wrapper(stats_cross_cv,cond,avg_cross_trials)
% WELCHTTEST_WRAPPER  Welch's t-test comparing within vs cross-condition r-values.
%   (Legacy function - predates the cluster-based permutation test pipeline.)
%
%   Tests two comparisons using independent-samples Welch t-tests:
%     wttf: R_ff (train fast, test fast) vs R_fs (train fast, test slow)
%     wtts: R_ss (train slow, test slow) vs R_sf (train slow, test fast)
%
%   Note: this function uses compile_rvals in a way that relies on specific
%   positional outputs that no longer match its current return signature.
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


%Maris, Eric, and Robert Oostenveld. "Nonparametric Statistical Testing of 
% EEG- and MEG-Data.” Journal of Neuroscience Methods 164, no. 1 (2007): 
% 177–90. https://doi.org/10.1016/j.jneumeth.2007.03.024.
