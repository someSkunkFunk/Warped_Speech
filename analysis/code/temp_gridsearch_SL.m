function [best_params, best_costs]=gridsearch_SL(eeg_trials,stim_trials, ...
    f_nat,sl_config)
%TODO: grid-loss function visualization
%TODO: consider possible benefit of maximizing correlation instead of
%minimizing SSR
% [best_params, best_r]=gridsearch_SL(eeg,stim,fs)
% eeg_trials: {1 x trials}-> [time x chns] SINGLE SUBJECT!
% stim_trials: {trials x 1}->[time x 1] SINGLE SUBJECT!
% f_nat: sl model natural frequency in Hz -- we set this manually outside
% the function rather than optimizing since related to hypothesis
% best_params: [electrodes x params (lambda, gamma, k)]  
% sl_config.n_starts: number of starting points for lsqnonlin
% --- mask subset of electrodes to use for model-fitting ---

eeg_trials=cellfun(@(x) x(:,sl_config.fit_chns), eeg_trials, ...
    'UniformOutput',false);
if sl_config.fit_on_avg_chns
    % average EEG across selected channels to fit one model instead of
    % nchns
    eeg_trials=cellfun(@(x) mean(x,2),eeg_trials, 'UniformOutput',false);
end

%  --- set up COARSE parameter grid ---
%TODO: flag when best_params occur near grid edge
grid_len=8; % number of points in grid
lambda_vec=linspace(0.01,1,grid_len); %TODO: see if this is a plausible range or need to extend
% gamma_vec=linspace(0.01,2,grid_len); %TODO: ^
rL_vec=linspace(0.001,.1,grid_len); % search speace over a given limit cycle radius instead of all gammas directly
k_vec=linspace(0.001, .01, grid_len); %TODO: ^ (check z-scored EEG rms range to see if appropriate to assume 1 max??) 

% grid length doesn't necessarily have to be the same
grid_size=numel(lambda_vec)*numel(rL_vec)*numel(k_vec);


n_chns=size(eeg_trials{1},2);
best_params=nan(n_chns,3);
best_costs=inf(n_chns,1);
% normalize EEG & stim RMS to restrain k-range
switch sl_config.normalize_envs
    case 'rms-global'
        disp('normalizing stim/eeg by to unit RMS globally')
        %[time x 1]:
        stim_trials_concat_=cell2mat(stim_trials);
        stim_global_rms_=rms(stim_trials_concat_);
        stim_trials=cellfun(@(x)x./stim_global_rms_,stim_trials, ...
            'UniformOutput',false);
        
        % [time x chns]:
        eeg_trials_concat_=cell2mat(cellfun(@(x) x, ...
            eeg_trials,'UniformOutput',false)'); % cellfun needed here for cat
        eeg_global_rms_=rms(eeg_trials_concat_);
        % RMS operates along columns, which works because we
        % have channels as columns -- broadcast divide
        eeg_trials=cellfun(@(x) x./eeg_global_rms_, ...
            eeg_trials,'UniformOutput',false);
        
        clear stim_trials_concat_ stim_global_rms_ eeg_trials_concat_ eeg_global_rms_
    otherwise
        error('normalization using %s not configured in gridsearch.', ...
            sl_config.normalize_envs)

end


%--- loop gridsearch over trials, electrodes

for ch=1:n_chns
%---- coarse gridsearch ----
fprintf('Begin coarse Gridsearch for chn %d of %d\n',ch,n_chns)
costs=inf(numel(lambda_vec),numel(rL_vec),numel(k_vec));
eeg_trials_singleChn=cellfun(@(x) x(:,ch), eeg_trials,'UniformOutput',false);
tic
gs_counter_=0;
for li=1:numel(lambda_vec)
    for ri=1:numel(rL_vec)
        for ki=1:numel(k_vec)
            gs_counter_=1+gs_counter_;
            fprintf('Gridpoint %d of %d...\n',gs_counter_, ...
                grid_size)
            p=[lambda_vec(li), lambda_vec(li)./(rL_vec(ri))^2, k_vec(ki)];
            resid=multiTrial_residuals(p,f_nat,stim_trials,eeg_trials_singleChn,sl_config);
            costs(li,ri,ki)=sum(resid.^2);
        end
    end
end
toc
fprintf('Begin multistart lsq for chn %d of %d\n',ch,n_chns)
tic
% --- extract top N starting points ---
costs_flat=costs(:);
[~,sort_idx]=sort(costs_flat);
[li,ri,ki]=ind2sub(size(costs),sort_idx(1:sl_config.n_starts));
% best_cost=inf;
% best_params_ch=nan(3,1);
ub=[max(lambda_vec), max(rL_vec), max(k_vec)];
lb=[min(lambda_vec), min(rL_vec), min(k_vec)];
opts=optimoptions('lsqnonlin','Display','off',...
    'FunctionTolerance',1e-6,'MaxIterations',300);
for s=1:sl_config.n_starts
    lam0=lambda_vec(li(s));
    rL0=rL_vec(ri(s));
    k0=k_vec(ki(s));
    p0=[lam0,lam0./(rL0)^2,k0];
    try
        [p_opt, cost]=lsqnonlin( ...
            @(p) multiTrial_residuals(p,f_nat,stim_trials, ...
            eeg_trials_singleChn,sl_config), ...    
            p0,lb,ub,opts);
        if cost<best_costs(ch)
            best_costs(ch)=cost;
            best_params(ch,:)=p0;
        end
    catch
        disp(p0)
        warning('lsqnonlin failed with p0 above ^, moving on to next startpoint.')
        continue
    end
end
toc

end



% check for parameters at grid edges
for ch=1:n_chns
    best_params_ch=best_params(ch,:)'; % [param x 1]
    % convert back to limit cycle radius
    best_params_ch(2)= sqrt(best_params_ch(1)/best_params_ch(2));
    grid_lims=[lambda_vec(1), lambda_vec(end); ...
        rL_vec(1), rL_vec(end); k_vec(1), k_vec(end)]; %[param x 2]
    if any(best_params_ch==grid_lims,"all")
        disp('best_params:')
        disp(best_params_ch)
        disp('coincide with grid lims:')
        disp(grid_lims)
        error('expand grid and re-do gridsearch')
    end
end
end