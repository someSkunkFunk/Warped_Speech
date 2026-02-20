% Microstate analysis
% assumes avg_models, experiment_conditions, ind_models
% have been constructed by running 
% plot_trfs script prior to this

% restrict TRFs to -100ms to 500ms
microstate_analysis.t_range=[-100, 500];
time=avg_models(1).t;
m_time=time>=microstate_analysis.t_range(1)& ...
    time<=microstate_analysis.t_range(2);
time=time(m_time);
%% "average-reference" trfs
n_chns=length(chanlocs);
n_time=length(time);
% conditions x time x chns
ar_trfs_grand=nan([numel(experiment_conditions), n_time, n_chns]);

for cc=1:numel(experiment_conditions)
    W=squeeze(avg_models(1,cc).w);
    % restrict time range for analysis
    W=W(m_time,:);
    ar_trfs_grand(cc,:,:)=W-mean(W,1);
end
disp('avg referencing + grand avg trfs done.')
% repeat for single-subjects.
% subj x conditions x time x chns
ar_trf_subj=nan(horzcat(n_subjs,size(ar_trfs_grand)));
for ss=1:n_subjs
    for cc=1:numel(experiment_conditions)
        W=squeeze(ind_models(ss,cc).w);
        % restrict time range
        W=W(m_time,:);
        ar_trf_subj(ss,cc,:,:)=W-mean(W,1);
    end
end
disp('avg referencing + aggregating subj trfs done.')
%% compute pairwise DISS between each condition's grand average trfs


cond_pairs=nchoosek(1:numel(experiment_conditions),2);
n_pairs=length(cond_pairs);
global_diss=nan(n_pairs,n_time);
for cp=1:length(cond_pairs)
    for tt=1:numel(time)
        global_diss(cp,tt)=compute_diss(ar_trfs_grand(cond_pairs(cp,1),tt,:), ...
            ar_trfs_grand(cond_pairs(cp,2),tt,:));
    end
end
disp('obs DISS calculation done.')

%% do TANOVA
% not an analysis of variance -- just the name given by Murray et al to a
% non-parametric randomization test
n_perm=1000;
% preallocate DISS distribution array
perm_diss=nan(n_pairs,n_perm,n_time);
for cp=1:n_pairs
    fprintf('generating DISS null %d/%d\n',cp,n_pairs)
    c1=cond_pairs(cp,1);
    c2=cond_pairs(cp,2);

    
    for pp=1:n_perm
        fprintf('perm %d/%d\n',pp,n_perm)
        % preallocate permuted grand trf
        perm_grand_c1=zeros(n_time,n_chns);
        perm_grand_c2=zeros(n_time,n_chns);
        for ss=1:n_subjs
            % permute subject-level condition labels
            if rand>0.5
                perm_grand_c1=perm_grand_c1+ar_trf_subj(ss,c1,:,:);
                perm_grand_c2=perm_grand_c2+ar_trf_subj(ss,c2,:,:);
            else
                perm_grand_c2=perm_grand_c2+ar_trf_subj(ss,c1,:,:);
                perm_grand_c1=perm_grand_c1+ar_trf_subj(ss,c2,:,:);
            end
        end
        % calculate permuted grand avg trfs
        perm_grand_c1=perm_grand_c1/n_subjs;
        perm_grand_c2=perm_grand_c2/n_subjs;
  
        % get DISS values for current permutation
        for tt=1:n_time
            perm_diss(cp,pp,tt)=compute_diss(perm_grand_c1(tt,:),perm_grand_c2(tt,:));
        end
    end

end
%% compute p-val


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
