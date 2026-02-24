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
tanova.param.experiment=script_config.experiment;
% tanova.param.t_range=clustering.t_range;
n_chns=length(chanlocs);
n_time=length(time);
tanova.result.time=time;
global boxdir_mine
tanova_out_pth=fullfile(boxdir_mine,'analysis','tanova',[tanova.param.experiment '.mat']);
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
    clear c1 c2 tanova_m_
end
%% part 2: microstate clustering
% "what are the distinct topographic patterns in the data?"
% k-means clustering based on nearby channel correlations
% desired output: k_components x 2 cell with start, end times
clustering=struct('param',[],'result',[]);
% how to choose max number of clusters worth trying? maybe looking at KL
% curve?
clustering.param.k_clusters=2:10; 
clustering.param.t_range=tanova.param.t_range;
clustering.param.n_restarts=10; %note: use 100 ultimately...
% concatenate all trfs -- lumping conditions together
% still use average reference
% note: permute needed to keep waveform times from being shuffled around by
% conditions...
trf_concat=reshape(permute(tanova.result.ar_trfs_grand,[2 1 3]),[], n_chns);
% calculate lumped-GFP
gfp_concat=rms(trf_concat,2);
%% run k-means clustering
for k=clustering.param.k_clusters
    fprintf('')
end

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
