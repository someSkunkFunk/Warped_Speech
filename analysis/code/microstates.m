% Microstate analysis

%% part 1: TANOVA
% "do conditions produce statistically different scalp topographies, and at which time points?"
% assumes avg_models, experiment_conditions, ind_models
% have been constructed by running 
% plot_trfs script prior to this
tanova=struct('param',[],'result',[]);
% restrict TRFs to -100ms to 500ms
% only 
tanova.param.t_range=[-100, 500];
time=avg_models(1).t;
m_time=time>=tanova.param.t_range(1)& ...
    time<=tanova.param.t_range(2);
time=time(m_time);

tanova.param.overwrite=true; % avoid re-running permutations
% tanova.param.experiment=script_config.experiment;
% tanova.param.t_range=clustering.t_range;
n_chns=length(chanlocs);
n_time=length(time);
tanova.result.time=time;
global boxdir_mine
tanova_out_pth=fullfile(boxdir_mine,'analysis','tanova',[script_config.experiment '.mat']);
if exist(tanova_out_pth,'file')==0 || tanova.param.overwrite
    disp('no file found -- running TANOVA.')
    %% "average-reference" trfs
    % conditions x time x chns
    ar_trfs_grand=nan([numel(experiment_conditions), n_time, n_chns]);
    
    for cc=1:numel(experiment_conditions)
        W=squeeze(avg_models(1,cc).w);
        % restrict time range for analysis
        W=W(m_time,:);
        ar_trfs_grand(cc,:,:)=W-mean(W,2);
    end
    disp('avg referencing + grand avg trfs done.')
    % repeat for single-subjects.
    % subj x conditions x time x chns
    ar_trfs_subj=nan(horzcat(n_subjs,size(ar_trfs_grand)));
    for ss=1:n_subjs
        for cc=1:numel(experiment_conditions)
            W=squeeze(ind_models(ss,cc).w);
            % restrict time range
            W=W(m_time,:);
            ar_trfs_subj(ss,cc,:,:)=W-mean(W,2);
        end
    end
    disp('avg referencing + aggregating subj trfs done.')
    tanova.result.ar_trfs_grand=ar_trfs_grand;
    tanova.result.ar_trfs_subj=ar_trfs_subj;
    %% compute pairwise DISS between each condition's grand average trfs
    
    
    cond_pairs_=nchoosek(1:numel(experiment_conditions),2);
    n_pairs=length(cond_pairs_);
    diss_obs=nan(n_pairs,n_time);
    for cp=1:length(cond_pairs_)
        for tt=1:numel(time)
            diss_obs(cp,tt)=compute_diss(ar_trfs_grand(cond_pairs_(cp,1),tt,:), ...
                ar_trfs_grand(cond_pairs_(cp,2),tt,:));
        end
    end
    disp('obs DISS calculation done.')
    tanova.result.diss_obs=diss_obs;
    clear diss_obs 
    %% do TANOVA
    % not an analysis of variance -- just the name given by Murray et al to a
    % non-parametric randomization test
    tanova.param.n_perm=1000;
    % preallocate DISS distribution array
    diss_perm=nan(n_pairs,tanova.param.n_perm,n_time);
    for cp=1:n_pairs
        fprintf('generating DISS null %d/%d\n',cp,n_pairs)
        c1=cond_pairs_(cp,1);
        c2=cond_pairs_(cp,2);
    
        
        for pp=1:tanova.param.n_perm
            fprintf('perm %d/%d\n',pp,tanova.param.n_perm)
            % preallocate permuted grand trf
            grand_perm_c1_=zeros(n_time,n_chns);
            grand_perm_c2_=zeros(n_time,n_chns);
            for ss=1:n_subjs
                % permute subject-level condition labels
                if rand>0.5
                    grand_perm_c1_=grand_perm_c1_+squeeze(ar_trfs_subj(ss,c1,:,:));
                    grand_perm_c2_=grand_perm_c2_+squeeze(ar_trfs_subj(ss,c2,:,:));
                else
                    grand_perm_c2_=grand_perm_c2_+squeeze(ar_trfs_subj(ss,c1,:,:));
                    grand_perm_c1_=grand_perm_c1_+squeeze(ar_trfs_subj(ss,c2,:,:));
                end
            end
            % calculate permuted grand avg trfs
            grand_perm_c1_=grand_perm_c1_/n_subjs;
            grand_perm_c2_=grand_perm_c2_/n_subjs;
      
            % get DISS values for current permutation
            for tt=1:n_time
                diss_perm(cp,pp,tt)=compute_diss(grand_perm_c1_(tt,:),grand_perm_c2_(tt,:));
            end
            clear grand_perm_c1_ grand_perm_c2_
        end
        clear c1 c2
    end
    disp('saving tanova results.')
    tanova.result.diss_perm=diss_perm;
    clear diss_perm ar_trfs_grand ar_trfs_subj
    % compute p-val
    % add singleton for broadcasting
    diss_obs_=reshape(tanova.result.diss_obs, [n_pairs,1, n_time]);
    tanova.result.p=squeeze(mean(tanova.result.diss_perm>=diss_obs_,2));
    tanova.result.cond_pairs=cond_pairs_;
    save(tanova_out_pth,"tanova")
    fprintf('saved %s\n',tanova_out_pth);
    clear diss_obs_ cond_pairs_
else
    % load pre-calculated results
    disp('file found -- preloading tanova results.')
    load(tanova_out_pth,'tanova')
end
%% plot p-value -- diffence in topos across conditions?
tanova.param.alpha=0.05; % significance threshold
tanova.param.min_samples=3; % at least this many samples to be considered significant
tanova_fig=[];
tanova_fig.threshold=0.9;
tanova_fig.ylims=[0.9 1.0];

for cp=1:length(tanova.result.cond_pairs)
    c1=tanova.result.cond_pairs(cp,1);
    c2=tanova.result.cond_pairs(cp,2);
    figure
    % remove subthreshold values to avoid weird shapes from fill
    p_inv=1-tanova.result.p(cp,:);
    tanova_m_=p_inv>tanova_fig.threshold;
    p_inv=p_inv.*tanova_m_;
    % 
    mask_diff=[0, diff(tanova_m_)];
    run_starts=find(mask_diff==1);
    run_ends=find(mask_diff==-1);
    % if start or end clipped -- extend length of run to edge of analysis
    % time window
    if length(run_starts)-length(run_ends)==1
        run_ends=[run_ends, length(p_inv)];
    elseif length(run_ends)-length(run_starts)==1
        run_starts=[1, run_starts];
    end
    if length(run_starts)~=length(run_ends)
        warning('mismatch here')
    end
    for rr=length(run_starts)
        % remove overly-short time windows
        if run_ends(rr)-run_starts(rr)+1 > tanova.param.min_samples
            rs=run_starts(rr);
            re=run_ends(rr);
            fill(tanova.result.time(rs:re), p_inv(rs:re),'r')
            hold on
        end
    end
    tstr=sprintf('TANOVA: %s vs %s',experiment_conditions{c1}, ...
        experiment_conditions{c2});
    title(tstr)
    xlabel('time (ms)')
    ylabel('1-p value')
    ylim(tanova_fig.ylims)
    xlim(tanova.param.t_range)
    yticks([0.9 0.95 1.0])
    hold off
    clear c1 c2 tanova_m_ run_starts run_ends
end
%% part 2: microstate clustering
% "what are the distinct topographic patterns in the data?"
% k-means clustering based on nearby channel correlations
% desired output: k_components x 2 cell with start, end times
clustering=struct('param',[],'result',[]);
% how to choose max number of clusters worth trying? maybe looking at KL
% curve?
clustering.param.ks=2:15;
clustering.param.t_range=tanova.param.t_range;
clustering.param.n_restarts=100; %note: use 100 ultimately...
clustering.param.overwrite=true;
% clustering.param.experiment=tanova.param.experiment;
% concatenate all trfs -- lumping conditions together
% still use average reference
% note: permute needed to keep waveform times from being shuffled around by
% conditions...
% i.e. [conditions x time x chns] -> [time x conditions x chns]
%   (-> [time*conditions x chns])
trfs_concat=reshape(permute(tanova.result.ar_trfs_grand,[2 1 3]),[], n_chns);
% calculate lumped-GFP
gfp_concat=rms(trfs_concat,2);
% GFP-normalize each timepoint map before clustering
% ensures clustering driven by topography, not strength
trfs_normed=trfs_concat./gfp_concat;
%% run k-means clustering
n_maps_total=size(trfs_normed,1); % n_conditions * n_time
% preallocate GEV (global explained variance)
clustering_out_pth=fullfile(boxdir_mine,'analysis','kmeans_clustering',[script_config.experiment '.mat']);
if exist(clustering_out_pth,'file')==0 || clustering.param.overwrite
    clustering.result.GEV=nan(length(clustering.param.ks),1);
    clustering.result.labels=cell(length(clustering.param.ks),1);
    clustering.result.centroids=cell(length(clustering.param.ks),1);
    
    for k_idx=1:length(clustering.param.ks)
        k=clustering.param.ks(k_idx);
        fprintf('clustering, k=%d...\n',k)
    
        [labels, centroids]=kmeans(trfs_normed, k, ...
            'Replicates',clustering.param.n_restarts,...
            'Distance','cosine' ...
            );
        % labels: [n_maps_total-by-1] centroids: [k-by-chns]
        % note: consider changing MaxIter using statset if non-convergence is
        % suspected... should primarily rely on reinilization though
        
        % compute GEV for this k solution
        % GEV = sum(GFP(t)^2 * C(t)^2 / sum(GFP(t)^2)
        % where C(t): spatial correlation at timepoint t --
        %                   corr(trf_concat(t,:), centroids(labels(t),:))
        % C(u,v) = (u'*v) / (||u|| * ||v||)
        C=sum(trfs_normed.*centroids(labels,:), 2) ./ ...
        (sqrt(sum(trfs_normed.^2,2)).*sqrt(sum(centroids(labels,:).^2,2)));
    
        clustering.result.GEV(k_idx)=sum(gfp_concat.^2.*C.^2)/sum(gfp_concat.^2);
        clustering.result.labels{k_idx}=labels;
        clustering.result.centroids{k_idx}=centroids;
        clear labels centroids
    end
    save(clustering_out_pth,'clustering');
    fprintf('saved clustering results to %s\n',clustering_out_pth)
else
    load(clustering_out_pth,'clustering')
end
%% plot GEV to inspect the elbow
figure;
plot(clustering.param.ks, clustering.result.GEV, 'o-')
xlabel('Number of clusters')
ylabel('GEV')
title('Global explained variance by cluster solution')

figure;
plot(clustering.param.ks(2:end), diff(clustering.result.GEV), 'o-')
xlabel('Number of clusters')
ylabel('delta GEV')
title('Incremental GEV gain')
%% choose optimal k (KL criterion)
% Krzanowski-Lai criterion
% Dispersion (W) trends towards 0 as the quality of the clustering results

W=nan(length(clustering.param.ks),1);
for k_idx=1:length(clustering.param.ks)
    k=clustering.param.ks(k_idx);
    labels=clustering.result.labels{k_idx};
    centroids=clustering.result.centroids{k_idx};

    W_q=0;
    for r=1:k
        % find all maps assigned to this cluster
        cluster_maps=trfs_normed(labels==r,:);
        % number of maps for cluster r
        % note: might be slow... could do without enumerating pairs
        % explicitly perhaps?
        n_r=size(cluster_maps,1);

        % Sum of pair-wise distance between all maps of a given cluster r
        cluster_pairs=nchoosek(1:size(cluster_maps,1),2);
        Dr=sum(sum((cluster_maps(cluster_pairs(:,1),:)-cluster_maps(cluster_pairs(:,2),:)).^2,2));
        W_q=W_q+Dr./(2*n_r);
    end
    W(k_idx)=W_q;
    clear W_q labels centroids
end
disp('dispersion calculation done.')
%% calculate KL criterion, find optimal number of clusters
% normalize W by k^(2/n_chns) per Murray et al
M_q = W .* (clustering.param.ks'.^(2/n_chns));
% compute curvature
d_q = diff(M_q);
KL_q = abs(d_q(1:end-1) ./ d_q(2:end));
warning('not saving KL_q to file below...')
clustering.result.KL_q=KL_q; %todo: other things worth saving...?
[~,opt_k_idx]=max(KL_q);
templates=clustering.result.centroids{opt_k_idx}; % [4-by-chns]
labels_grand=clustering.result.labels{opt_k_idx}; % [n_maps_total-by-1]
%% plot KL criterion
figure;
plot(clustering.param.ks(2:end-1), KL_q, 'o-')
xlabel('Number of clusters')
ylabel('KL criterion')
title('Krzanowski-Lai criterion')
%% component definition from clustering
% based on lalor et al 2009: define component windows from template map
% transitions in the grand average TRF
components_out_pth=fullfile(boxdir_mine,'analysis','components',[script_config.experiment '.mat'])
components=struct('param',[],'result',[]);
components.param.k=opt_k_idx;
components.param.overwrite=true;
if exist(components_out_pth,"file")==0 || components.param.overwrite
    
    % get labels for concatenated grand average
    % labels: [time*conditions x 1] (cuz trf_concat)
    % -> [time x conditions]
    labels_by_cond=reshape(labels_grand,n_time,numel(experiment_conditions));
    
    % for condition-invariant boundaries, find dominant template at each
    % timepoint across conditions by majority vote
    labels_concat_time=mode(labels_by_cond,2); %[n_time x 1]
    
    % apply minimum duration constraint
    % find runs of same label
    components.param.min_duration_samples=3;
    run_starts=find(diff([0; labels_concat_time])~=0);
    run_ends=[run_starts(2:end)-1; n_time];
    run_lengths=run_ends-run_starts+1;
    
    if any(run_lengths<components.param.min_duration_samples)
        % merge with neighboring component
        for rr=1:length(run_starts)
            if run_lengths(rr)<components.param.min_duration_samples
                % replace short segment with label of preceding segment
                if run_starts(rr)>1
                    labels_concat_time(run_starts(rr):run_ends(rr))= ...
                        labels_concat_time(run_starts-1);
                    % run_starts(rr)=run_starts(rr-1);
                    % run_lengths(rr)=run_ends(rr)-run_starts(rr)+1;
                else
                    labels_concat_time(run_starts(rr):run_ends(rr))= ...
                        labels_concat_time(run_ends(rr)+1);
                    % run_starts(rr+1)=run_starts(rr);
                    % run_lengths(rr+1)=run_ends(rr+1)-run_starts(rr+1)+1;
                end
            end
        end
    end
    clear run_ends  run_starts
    
    %find component boundaries
    transitions=find(diff([0; labels_concat_time])~=0);
    comp_starts=[1; transitions+1];
    comp_ends=[transitions; n_time];
    comp_labels=labels_concat_time(comp_starts);
    
    n_components=length(comp_starts);
    
    fprintf('found %d components\n', n_components);
    components.result.starts=time(comp_starts);
    components.result.ends=time(comp_ends);
    %preallocate topos for each component
    components.result.topos=nan(n_components,n_chns);
    
    % average grand trf topography within each component window
    % use mean across conditions...?
    grand_mean_trf=squeeze(mean(tanova.result.ar_trfs_grand,1)); %[time x chns]
    
    for cc=1:n_components
        window=comp_starts(cc):comp_ends(cc);
        components.result.topos(cc,:)=mean(grand_mean_trf(window,:),1);
        clear window
    end
    disp('component topos evaluated.')
    save(components_out_pth,"components");
    fprintf('saved components to %s\n',components_out_pth)
else
    load(components_out_pth,"components");
end
%% plot Lalor et al 2009 figure 4 using these component windows
% caution: component_windows variable defined in plot_trfs could cause
% confusion here.
%%
function diss=compute_diss(u,v)
    % per Murray et al 2008, "global dissimilarity (DISS) is an index of
    % configuration differences between two electric fields, independent of
    % their strength... equals the square root of the mean of the squared
    % differences between the potentials measured at each electrode vs 
    % the average reference"
    u=squeeze(u);
    v=squeeze(v);
    % assumes u,v are average-referenced
    % normalize each map by its GFP
    u_norm=u/norm(u);
    v_norm=v/norm(v);
    diss=rms(u_norm-v_norm);
end

% references
% Murray, Micah M., Denis Brunet, and Christoph M. Michel. 
% “Topographic ERP Analyses: A Step-by-Step Tutorial Review.” Brain 
% Topography 20, no. 4 (2008): 
% 249–64. https://doi.org/10.1007/s10548-008-0054-5.

% Lalor, Edmund C., Alan J. Power, Richard B. Reilly, and John J. Foxe. 
% “Resolving Precise Temporal Processing Properties of the Auditory 
% System Using Continuous Stimuli.” Journal of Neurophysiology 102, 
% no. 1 (2009): 349–59. https://doi.org/10.1152/jn.90896.2008.

