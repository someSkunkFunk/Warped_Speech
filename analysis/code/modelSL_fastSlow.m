% model SL response to fast-slow stimuli
% first optimize parameters using gridsearch on existing data
% then simulate responses from optimal model and derive trf from it
%% --- GENERAL SETTINGS ---
subj=2; % DO SINGLE SUBJECT AT A TIME
cond_nms={'fast','original','slow'};
sl_config=[];
sl_config.SKIP_DATA=false; % if true -> skip parts of script that require
sl_config.RELOAD=false; % if need to load data, first check if it's already been loaded
% data because they are slow

sl_config.init='limit cycle'; %limit cycle or rand (uniform [-1,1])
sl_config.fs=128;
sl_config.tmax_zero_input=60; % time limit in seconds for undriven input simulation
sl_config.trials_to_plot=[10]; % look at simulated response for trials specified here
% --- model options --- 
% 'env' : envelope-coupled by some constant
% 'reset': phase-reset model 
sl_config.model='env'; 
sl_config.fit_trfs=true;
% normalize_envs options: 
% 'doelling' - same thing as Doelling et al 2019
% 'range' - all envelopes normalized between 0,1
% 'rms' - compute global RMS of stimuli and normalize them by that value
% NOTE: RMS IS WHAT GRIDSEARCH USES DURING OPTIMIZATION SO SHOULD BE USED
% DURING SIMULATION FOR TRF FITTING TO BE CONSISTENT
sl_config.normalize_envs='rms-global';
sl_config.demo_klim=false;  
% 1:120 to simulate all of them, shorten to speed up trf training time
sl_config.sim_trials=[10]; 
% find parameter by optimizing if true, otherwise use hard-coded (in future
% should look up saved optimized parameters -- or maybe that can be part of
% optimization script also)
sl_config.optimize_sl=true;

sl_config.match_syllable_frequency=false;
sl_config.avg_syllable_rate=4; % Hz
% sl_config.rms_normalize=true;
% 
sl_config.use_solver='RK4';
sl_config.fit_chns=[54 55 56 61 62 63 106 107 108 115 116 117];
sl_config.fit_on_avg_chns=true;
%% select SL parameters


if sl_config.match_syllable_frequency
    %TODO: match average syllable frequency in each condition -- then make code downstream account for variable f_nat
    rate_scalings_=[2/3 1 3/2];
    sl_param=cell2struct(cell(size(cond_nms)),cond_nms,2);
    for cc=1:length(cond_nms)
        sl_param.(cond_nms{cc}).f_nat=sl_config.avg_syllable_rate/rate_scalings_(cc);
    end
    clear rate_scalings_
else
    sl_param=[];
    sl_param.f_nat=sl_config.avg_syllable_rate; % in Hz -> converted to radians when running model
end
switch sl_config.model
    case 'env'
        if sl_config.optimize_sl && ~sl_config.SKIP_DATA

            if sl_config.RELOAD
                disp('loading all subj stim,resp')
                tic
                
                [stim_trials,...
                    eeg_trials,...
                    cond_trials]=load_fastSlow_data(subj);
            
                disp('all subj data loaded')
                
                toc
                
            elseif ~sl_config.RELOAD&&(~exist('stim_trials', 'var')||~exist('eeg_trials','var'))
                error('MISSING EITHER STIM OR VAR -- RESET RELOAD TO TRUE')
            end
            %% ----optimize_sl_env----
            
            fprintf('running gridsearch, subj %d\n', subj)

            if sl_config.match_syllable_frequency
                best_params=cell2struct(cell(size(cond_nms)),cond_nms,2);
                best_r_sl=cell2struct(cell(size(cond_nms)),cond_nms,2);
                for cc=1:length(cond_nms)
                    fprintf('cond: %s...\n',cond_nms{cc})
                    cond_mask_=cc==cond_trials;
                    [best_params.(cond_nms{cc}), best_r_sl.(cond_nms{cc})]=gridsearch_SL(eeg_trials(cond_mask_), ...
                    stim_trials(cond_mask_),sl_param.(cond_nms{cc}).f_nat, sl_config);
                end
            else
                [best_params, best_r_sl]=gridsearch_SL(eeg_trials, ...
                    stim_trials,sl_param.f_nat, sl_config);
            end
            %% save optimized params
            global boxdir_mine
            if sl_config.match_syllable_frequency
                freq_str_='fm'; % frequency-matched
            else
                freq_str_='ff'; % fixed-frequency
            end
            optim_param_path=fullfile(boxdir_mine, 'analysis',...
                    'sl_env',...
                    sprintf('sl_env_%s_s%02d.mat',freq_str_,subj));
            fprintf('saving optimization params to %s...\n',optim_param_path)
            save(optim_param_path,'sl_config','best_params','best_r_sl')
            disp('done'), clear freq_str_
        else
            warning('Setting SL parameters with hard-coded parameters.')
            % optimal parameters wei ching previously gave 
            sl_param.lambda=0.1;
            sl_param.gamma=13.83;
            sl_param.k=80;
            % sl_param.lambda=.01;
            % sl_param.gamma=1;
            % r_limit_cycle=sqrt(sl_param.lambda/sl_param.gamma);
            % sl_param.k=2*rs;
            % sl_param.k=rs/2;
        end

    case 'reset'
        if sl_config.optimize_sl
            optimize_sl_reset
        else
            warning('hard coded parameters not yet specified.')
        end
end
%% set params based on optimization result
if sl_config.optimize_sl
    if sl_config.match_syllable_frequency
        for cc=1:length(cond_nms)
            sl_param.(cond_nms{cc}).lambda=best_params.(cond_nms{cc})(1);
            sl_param.(cond_nms{cc}).gamma=best_params.(cond_nms{cc})(2);
            sl_param.(cond_nms{cc}).k=best_params.(cond_nms{cc})(3);
        end
    else
        sl_param.lambda=best_params(1);
        sl_param.gamma=best_params(2);
        sl_param.k=best_params(3);
    end
end


%% UNFORCED MODEL CHARACTERIZATION
%% run model without input
switch sl_config.model
    case 'env'
        if sl_config.match_syllable_frequency
            for cc=1:length(cond_nms)
                sl_param_=sl_param.(cond_nms{cc});
                [t_nostim(cc,:), sl_nostim(cc,:,:)]=run_sl_env(sl_config,sl_param_);
                clear sl_param_
            end
        else
            [t_nostim, sl_nostim]=run_sl_env(sl_config,sl_param);
        end
    case 'reset'
        [t_nostim, sl_nostim]=run_sl_reset(sl_config,sl_param);
    otherwise
        error('config.model: %s', sl_config.model)
end

%% plot unforced model output
figure('Color','white')
if sl_config.match_syllable_frequency
    for cc=1:length(cond_nms)
        subplot(3,1,cc)
        plot(t_nostim(cc,:),sl_nostim(cc,:,1)), hold on
        plot(t_nostim(cc,:),sl_nostim(cc,:,2))
        title(sprintf('%s, $\\lambda$ = %0.3g, $\\gamma$ = %0.3g',cond_nms{cc}, ...
            sl_param.(cond_nms{cc}).lambda,sl_param.(cond_nms{cc}).gamma),'Interpreter','latex')
        legend('x', 'y')
        xlabel('time (s)')
        xlim([0 1])
        hold off
    end
    sgtitle(sprintf('SL-%s without stimulus',sl_config.model))

else
    plot(t_nostim,sl_nostim(:,1)), hold on
    plot(t_nostim,sl_nostim(:,2))
    title(sprintf('SL-%s without stimulus',sl_config.model))
    legend('x', 'y')
    xlabel('time (s)')
    xlim([0 1])
    hold off
end

%% phase-portrait of unforced model output
figure('Color', 'white')
plot(sl_nostim(:,1),sl_nostim(:,2))
axis equal
xlabel('x')
ylabel('y')
title(sprintf('Undriven %s phase portrait',sl_config.model))
%% arnold tongue
%% phase sensitivity curve?
%% FORCED RESPONSE CHARACTERIZATION
% goal: characterize model response to speech generally, not an individual
% envelope
%% --- SIMULATED  RESPONSES TO STIMULI ---
% note: previously the code from here on down was intended to look at
% expected SL response TRFs for different sets of candidate irreg stimuli 
% but now that we've decided upon a stimulus set to use we are using it to
% look at TRFs predicted by optimized sl modes on fast-slow data... which
% means the next time we want to go and do the same with the reg-irreg data
% we'll have to continue updating the code.
global boxdir_mine
sim_stim=stim_trials; % copy to variable to re-apply normalization etc
    switch sl_config.normalize_envs
    % normalize envelopes as in Doelling et al 2023
        case 'doelling'
            sim_stim=cellfun(@(x) norm_env_doelling(x),sim_stim,'UniformOutput',false);
        case 'range'
            % normalize to 0,1
            sim_stim=cellfun(@(x) normalize(x, 'range'),sim_stim,'UniformOutput',false);
        case 'rms'
            sim_stim_concat_=cell2mat(sim_stim);
            sim_stim_globalrms_=rms(sim_stim_concat_);
            sim_stim=cellfun(@(x) x./sim_stim_globalrms_, sim_stim,'UniformOutput',false);
            clear sim_stim_concat_ sim_stim_globalrms_
    end
    
    
    % run sl model for each trial's envelope OR demo model response to
    % sustained "maximum" DC stimulus for given k value ("klim")
    if sl_config.demo_klim
        % show deviation from limit cycle as stimulus exerts maximum force away
        % from limit cycle
        sl_responses=cell(1,2);
        [sl_responses{1,:}]=run_sl_env(sl_config,sl_param,ones(size(sim_stim{1})));
    else
        %%% --- SIMULATE OPTIMIZED SL MODEL RESPONSES ---
        sl_responses=cell(length(sim_stim),2);
        for ii=1:length(sim_stim)
            if sl_config.match_syllable_frequency
                cc=cond_trials(ii);
                sl_param_=sl_param.(cond_nms{cc});
            else
                sl_param_=sl_param;
            end
            
            fprintf('simulating %s-SL response for %d/%d\n',sl_config.model,ii,length(sim_stim))
            switch sl_config.model
                case 'env'
                    [sl_responses{ii,:}]=run_sl_env(sl_config,sl_param_,sim_stim{ii});
                case 'reset'
                    [sl_responses{ii,:}]=run_sl_reset(sl_config,sl_param_);
                otherwise
                    error('config.model: %s', sl_config.model)
            end
        clear sl_param_
        end

    end
%% look at "phase portrait," xy, and input-output plots for a particular trial
% TODO: add trial condition label to plots
if ~isempty(sl_config.trials_to_plot)
    for tt=1:length(sl_config.trials_to_plot)
        trial_idx=sl_config.trials_to_plot(tt);
        if sl_config.match_syllable_frequency
            sl_param_=sl_param.(cond_nms{cond_trials(trial_idx)});
        else
            sl_param_=sl_param;
        end
        r_limit_cycle_=sqrt(sl_param_.lambda/sl_param_.gamma);
        
        figure('Color', 'white')
        %note: time returned by model simulation should match fs samples of
        %stimulus because of how we defined tspan arg of ode45
        tstr_=sprintf('normalized stimulus envelope - trial %d (%s)', ...
            trial_idx,cond_nms{cond_trials(trial_idx)});
        plot(sl_responses{trial_idx,1},sim_stim{trial_idx})
        xlabel('time (s)')
        ylabel('S(t)')
        title(tstr_)

        %xy plot
        tstr_=sprintf('%s speech response - trial %d (%s)', ...
            sl_config.model,trial_idx,cond_nms{cond_trials(trial_idx)});
        figure('Color', 'white')
        plot(sl_responses{trial_idx,1},[sl_responses{trial_idx,2}, sim_stim{trial_idx}])
        title(tstr_)
        legend('x', 'y','env')
        xlabel('time (s)')
        xlim([min(t_nostim) max(t_nostim)])
        hold off
        

        % "phase portrait"
        figure('Color', 'white')
        plot(sl_responses{trial_idx,2}(:,1),sl_responses{trial_idx,2}(:,2))
        hold on
        thetas_=0:pi/100:2*pi;
        plot(r_limit_cycle_*cos(thetas_),r_limit_cycle_*sin(thetas_),'r--')
        hold off
        clear thetas_
        axis equal
        xlabel('x')
        ylabel('y')
        legend('SL simulation','limit cycle')
       
        tstr_=sprintf('%s speech-phase portrait - trial %d (%s)', ...
            sl_config.model,trial_idx,cond_nms{cond_trials(trial_idx)});
        title(tstr_)

        
        % input-output (just x)
        % NOTE: if time in sl doesn't line up with stimulus samples this is
        % not gonna correspond right...
        figure('Color', 'white')
        plot(sim_stim{trial_idx},sl_responses{trial_idx,2}(:,1))
        xlabel('S(t)')
        ylabel('x(t)')
        tstr_=sprintf('%s input-output - trial %d (%s)', ...
            sl_config.model,trial_idx,cond_nms{cond_trials(trial_idx)});
        title(tstr_)
        clear tstr_ sl_param_ 
    end
end
%% --- FIT TRFS ON SIMULATED RESPONSES ---
if sl_config.fit_trfs
    sl_trf_config=[];
    sl_trf_config.add_noise=false;
    sl_trf_config.optimize_lambda=false;
    sl_trf_config.lam_range=10.^(-3:8);
    % note: used 400ms previously but if we're looking at effects for syllables
    % up to 1000ms apart... think it makes sense to extend the range a bit
    sl_trf_config.cvtlims=[0, 400];
    sl_trf_config.trtlims=[-150,450];
    % add 1/f noise to simulated responses to make them "realistic"
    % Oganian et al 2023 did this by using firls on gaussian white noise
    % they also used an snr of 1/10... not sure why
    %note: leaving this out for now to look at "pure" model response behavior 
    if sl_trf_config.add_noise
        %TODO: add 1/f noise
    end
    sl_trf_config.env_normalization='std-global';
    switch sl_trf_config.env_normalization
        case 'std-global'
            stim_trials_concat_=cell2mat(stim_trials);
            stim_trials_stdglobal_=std(stim_trials_concat_);
            trf_env=cellfun(@(x) x./stim_trials_stdglobal_,stim_trials, ...
                'UniformOutput',false);
            clear stim_trials_concat_ stim_trials_stdglobal_
        otherwise
            error('invalid or unaccounted trf feature normalization: %s', ...
                sl_trf_config.env_normalization)
    end
    
    % optimize lambda via cv
    % NOTE: SL responses can be derived using a different normalization
    % than used during TRF fitting as long as the parameter optimization
    % used the same kind of normalization
    
    % z-score the sl-model output for trf fit to match what we did on eeg
    trf_response=cellfun(@(x) normalize(x(:,1)), sl_responses(:,2),'UniformOutput',false);
    if sl_trf_config.optimize_lambda
        stats_obs=mTRFcrossval(trf_env,trf_response,sl_config.fs,1, ...
            sl_trf_config.cvtlims(1),sl_trf_config.cvtlims(2),sl_trf_config.lam_range);
        % plot TRFs
        % plot lambda tuning curve, select best lambda
        r_tuning_curve=mean(stats_obs.r, 1);
        [~,max_lam_idx]=max(r_tuning_curve);
        sl_trf_config.best_lam=sl_trf_config.lam_range(max_lam_idx);
        figure('Color', 'white')
        plot(sl_trf_config.lam_range,r_tuning_curve)
        title('tuning curve')
        xlabel('\lambda')
        ylabel('r (crossvalidated)')
    else
        % seems unnecessary if no noise
        sl_trf_config.best_lam=0;
    end
    %%% --- TRAIN TRF ---
    
    % sort stimuli by condition
    trf_models=cell2struct(cell(size(cond_nms)),cond_nms,2);
    for cc=1:length(cond_nms)
        fprintf('training TRF for %s...\n',cond_nms{cc})
        cond_mask_=cond_trials==cc;
        trf_models.(cond_nms{cc})=mTRFtrain(trf_env(cond_mask_), ...
            trf_response(cond_mask_),sl_config.fs,1, ...
        sl_trf_config.trtlims(1),sl_trf_config.trtlims(2),sl_trf_config.best_lam);
    end
    disp('done.')

    
    %%% --- plot model-TRF ---
    cond_colors={'b','k','r'};
    figure('Color', 'white')
    for cc=1:length(cond_nms)
        if sl_config.match_syllable_frequency
            f_nat_=sl_param.(cond_nms{cc}).f_nat;
        else
            f_nat_=sl_param.f_nat;
        end
        plot(trf_models.(cond_nms{cc}).t, ...
            trf_models.(cond_nms{cc}).w, ...
            'Color',cond_colors{cc},...
            'DisplayName',sprintf('f_{nat}: %.1f Hz,  %s',f_nat_,cond_nms{cc}))
        hold on
        clear f_nat_
    end
    if sl_config.match_syllable_frequency
        freq_str_='frequency-matched';
    else
        freq_str_='frequency-fixed';
    end
    title(sprintf('%s-SL TRFs, %s',sl_config.model,freq_str_))
    legend()
    xticks(sl_trf_config.trtlims(1):100:sl_trf_config.trtlims(2))
    xlim([-100,400])
    xlabel('Time (ms)')
    ylabel('Amplitude (a.u.)')
end
%% quantify event-based phase concentration of output across all stimuli?
% to what end?
%


%% helpers
function [stim, eeg, cond]=load_fastSlow_data(subj)
    % subj: subject number used to define configs in trf_analysis_params
    script_config.show_tuning_curves=false;
    trf_analysis_params
    S_=load_checkpoint(preprocess_config);
    preprocessed_eeg=S_.preprocessed_eeg;
    eeg=preprocessed_eeg.resp;
    stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
    cond=preprocessed_eeg.cond;
end


function [best_params, best_costs]=gridsearch_SL(eeg_trials,stim_trials, ...
    f_nat,sl_config)
% [best_params, best_r]=gridsearch_SL(eeg,stim,fs)
% eeg_trials: {1 x trials}-> [time x chns] SINGLE SUBJECT!
% stim_trials: {trials x 1}->[time x 1] SINGLE SUBJECT!
% f_nat: sl model natural frequency in Hz -- we set this manually outside
% the function rather than optimizing since related to hypothesis
% best_params: [electrodes x params (lambda, gamma, k)]  

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
grid_len=15; % number of points in grid
lambda_vec=linspace(0.01,1,grid_len); %TODO: see if this is a plausible range or need to extend
% gamma_vec=linspace(0.01,2,grid_len); %TODO: ^
rL_vec=linspace(0.001,.1,grid_len); % search speace over a given limit cycle radius instead of all gammas directly
k_vec=linspace(0.001, .01, grid_len); %TODO: ^ (check z-scored EEG rms range to see if appropriate to assume 1 max??) 

% grid length doesn't necessarily have to be the same
grid_size=numel(lambda_vec)*numel(rL_vec)*numel(k_vec);

n_trials=length(eeg_trials);
n_chns=size(eeg_trials{1},2);

% best_rs=-Inf(n_chns,3);
best_costs=inf(n_chns);
best_params=nan(n_chns,3);
pred_trials=cellfun(@(x) nan(size(x)),eeg_trials,'UniformOutput',false);

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
tic

for ch=1:n_chns
%---- actual gridsearch start ----
fprintf('Begin Gridsearch for chn %d of %d\n',ch,n_chns);
costs=inf(numel(lambda_vec),numel(rL_vec),numel(k_vec));

gs_counter_=0;

for li=1:numel(lambda_vec)
    for ri=1:numel(rL_vec)
        for ki=1:numel(k_vec)
            gs_counter_=1+gs_counter_;
            fprintf('Gridpoint %d of %d...\n',gs_counter_, ...
                grid_size)
            % loop thru trials before concatenating predictions to get corr
            for tr=1:n_trials
                params=struct('lambda',lambda_vec(li), ...
                    'gamma',lambda_vec(li)./(rL_vec(ri))^2, ...
                    'k',k_vec(ki), ...
                    'f_nat',f_nat); %gets passed to gridsearch so not fit
                [~, sl_pred_trial]=run_sl_env(sl_config,params,stim_trials{tr});
                % assuming x is the observable response
                pred_trials{tr}(:,ch)=sl_pred_trial(:,1);
            end
            % concatenate trials to get corr -- cell2mat concatenates rows
            % like vertcat so need to transpose cell coming out of cellfun
            pred_concat=cell2mat(cellfun(@(x) x(:,ch), pred_trials,'UniformOutput',false)');
            eeg_concat=cell2mat(cellfun(@(x) x(:,ch), eeg_trials,'UniformOutput',false)');
            % r_pred=corr(pred_concat,eeg_concat);
            resid=eeg_concat-pred_concat;
            costs(li,ri,ki)=sum(resid.^2);
            if costs(li,ri,ki)<best_costs(ch)
                best_costs(ch)=costs(li,ri,ki);
                best_params(ch,:)=[lambda_vec(li) lambda_vec(li)./rL_vec(ri).^2, k_vec(ki)];
            end
            clear pred_concat eeg_concat
        end
    end
end
end
toc
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
function env_normed=norm_env_doelling(env_data)
% normalize stim as in Doelling et al 2023
env_bar=mean(env_data);
env_normed=(env_data-env_bar)/max(env_data-env_bar);
end
function [stim_env, event_times, trf_stim] = get_fast_slow_events()
    min_peak_height=0.01;
    global boxdir_mine
    envs_path = fullfile(boxdir_mine, 'stimuli', 'wrinkle', 'fastSlowEnvelopes128hz.mat');
    load(envs_path, 'env', 'fs');

    trf_stim.envs = env;
    trf_stim.fs   = fs;

    stim_env    = cell(size(env));
    event_times = cell(size(env));

    for cc = 1:size(env, 1)
        for ss = 1:size(env, 2)
            env_vec = env{cc,ss}(:);
            d_env   = [0; diff(env_vec)];

            % Event times = rising-edge peaks of envelope derivative
            [~, ev_t] = findpeaks(d_env, fs,'MinPeakHeight',min_peak_height);
            event_times{cc,ss} = ev_t;

            % Keep raw envelope as stimulus predictor for mTRF
            stim_env{cc,ss} = env_vec;
        end
    end
end

function x0=init_rand(n)
% initialize random starting condition;
% returns n values between [-1,1]
x0=-1*2*rand([n,1]);
end
function x0=init_limit_cycle(sl_param)
    % sample a random phase at limit cycle radius for stuart landau model
    lim_rad=sqrt(sl_param.lambda/sl_param.gamma);
    rand_theta=randn(1)*2*pi;
    x0=lim_rad.*[cos(rand_theta);sin(rand_theta)];
end
function [t, model_out]=run_sl_env(config,sl_param,s_data)
% [t, model_out]=run_sl_env(config,sl_param,s_data)
% arguments
%     config (1,1) struct
%     sl_param (1,1) struct
%     s_data (n,1) double = 0; stimulus as a vector -- 0 gives spontaneous
%     response 
% end
switch config.use_solver
    case 'ode45'
        if ~exist('s_data','var')
            tspan=0:1/config.fs:config.tmax_zero_input; % custom time limit for no input response
            input_fun=@(t,x) 0; % zero-function
        else
            % envelope was given - convert to "continuous time" for ode solver via
            % anonymous function with interp
            tspan=0:1/config.fs:(length(s_data)-1)/config.fs;
            input_fun=@(t,~) interp1(tspan,s_data,t,"pchip",0);
        end
    case 'RK4'
       if ~exist('s_data','var')
           tspan=0:1/config.fs:config.tmax_zero_input; % custom time limit for no input response
           s_data=zeros(size(tspan));
       else
           % just need to define tspan based on s_data
           tspan=0:1/config.fs:(length(s_data)-1)/config.fs;
       end
end
%TODO: how to flexibly switch between sl and wc models?
switch config.init
    case 'rand'
        x0=init_rand(2); 
    case 'limit cycle'
        % select random values already at the limit cycle
        x0=init_limit_cycle(sl_param);
    otherwise
        error('initialization config param not recognized.')
end
switch config.use_solver
    case 'ode45'
        % use ode solver to get sl model output:
        [t, model_out]=ode45(@(t,x) sl_model_cont(t,x,sl_param,input_fun),tspan,x0);
    case 'RK4'
        t=tspan;
        model_out=simulate_SL_RK4(sl_param,s_data,tspan,x0);
end
end

function model_out=simulate_SL_RK4(sl_param, s_data,t,x0)
    T=length(t);
    dt=t(2)-t(1);
    x=x0; % [x; y] - col vector
    
    model_out=nan(T,2); % [time x  x, y]
    for ii=1:T
        %RK4 estimation
        s=s_data(ii);
        k1=sl_model_step(x,sl_param,s);
        k2=sl_model_step(x+dt/2*k1,sl_param,s);
        k3=sl_model_step(x+dt/2*k2,sl_param,s);
        k4=sl_model_step(x+dt*k3,sl_param,s);
        x=x + (dt/6).*(k1 + 2*k2 + 2*k3 + k4);
        model_out(ii,:)=x;
    end
end
function dxdt=sl_model_step(x,sl_param,s)
    lambda=sl_param.lambda; gamma=sl_param.gamma;k=sl_param.k;
    omega=sl_param.f_nat*2*pi;
    r2=x(1)^2+x(2)^2;
    dxdt=nan(2,1);
    % x differential equation
    dxdt(1)=lambda*x(1)-omega*x(2)-gamma*r2*x(1)+k*s;
    % y differential equation
    dxdt(2)=lambda*x(2)+omega*x(1)-gamma*r2*x(2);
end

function dxdt=sl_model_cont(t, x, sl_param, s_fun)
% dxdt=sl_model(t, x, sl_param, Sfun)
% t: times to sample SL activity at
% x: col vector of x and y in SL equations
% sl_param.lambda: oscillatory "switch" parameter
% sl_param.f_nat: natural frequency in Hz (converted to rads/s here)
% sl_param.gamma: nonlinear gain control
% sl_param.k: coupling parameter
% s_fun: stimulus input specified as a function 

%TODO: define some default values
%unpack sl parameters
lambda=sl_param.lambda;
omega=sl_param.f_nat*2*pi;
gamma=sl_param.gamma;
k=sl_param.k;

r2=x(1)^2+x(2)^2;
S=s_fun(t,x);

dxdt=nan(2,1);
% x differential equation
dxdt(1)=lambda*x(1)-omega*x(2)-gamma*r2*x(1)+k*S;
% y differential equation
dxdt(2)=lambda*x(2)+omega*x(1)-gamma*r2*x(2);
end

%references
% Doelling, Keith B., Luc H. Arnal, and M. Florencia Assaneo. 
% “Adaptive Oscillators Support Bayesian Prediction in Temporal 
% Processing.” PLOS Computational Biology 19, no. 11 (2023): e1011669. 
% https://doi.org/10.1371/journal.pcbi.1011669.

% Oganian, Yulia, Katsuaki Kojima, Assaf Breska, et al. 
% “Phase Alignment of Low-Frequency Neural Activity to the Amplitude 
% Envelope of Speech Reflects Evoked Responses to Acoustic Edges, 
% Not Oscillatory Entrainment.” The Journal of Neuroscience 43, 
% no. 21 (2023): 3909–21. https://doi.org/10.1523/JNEUROSCI.1663-22.2023.

