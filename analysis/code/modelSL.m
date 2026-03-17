% model SL response
config=[];
config.init='limit cycle'; %limit cycle or rand (uniform [-1,1])
config.fs=128;
config.tmax_zero_input=60; % time limit in seconds for undriven input simulation
config.irreg_maxt=1000; % choose 1000, 750, or 500 ms max interval for irreg
config.plot_individual_trials=[10]; % look at simulated response for trials specified here
% --- model options --- 
% 'env' : envelope-coupled by some constant
% 'reset': phase-reset model 
config.model='env'; 
config.fit_trfs=false;
config.normalize_envs='range';
config.demo_klim=false;
config.sim_trials=[10]; % 1:120 to simulate all of them
%% select SL parameters

sl_param=[];
sl_param.f_nat=4; % in Hz -> converted to radians when running model
switch config.model
    case 'env'
        % optimal parameters wei ching previously gave 
        sl_param.lambda=0.1;
        sl_param.gamma=13.83;
        sl_param.k=80;
        % sl_param.lambda=.01;
        % sl_param.gamma=1;
        rs=sqrt(sl_param.lambda/sl_param.gamma);
        % sl_param.k=2*rs;
        % sl_param.k=rs/2;

    case 'reset'
        optimize_sl_reset
end

%% UNFORCED MODEL CHARACTERIZATION
%% run model without input
switch config.model
    case 'env'
        [t, sl_nostim]=run_sl_env(config,sl_param);
    case 'reset'
        [t, sl_nostim]=run_sl_reset(config,sl_param);
    otherwise
        error('config.model: %s', config.model)
end

%% plot unforced model output
figure('Color','white')
plot(t,sl_nostim(:,1)), hold on
plot(t,sl_nostim(:,2))
title(sprintf('%s without stimulus',config.model))
legend('x', 'y')
xlabel('time (s)')
% xlim([min(t) max(t)])
xlim([0 1])
hold off
%% phase-portrait of unforced model output
figure('Color', 'white')
plot(sl_nostim(:,1),sl_nostim(:,2))
axis equal
xlabel('x')
ylabel('y')
title(sprintf('Undriven %s phase portrait',config.model))
%% arnold tongue
%% phase sensitivity curve?
%% FORCED RESPONSE CHARACTERIZATION
% goal: characterize model response to speech generally, not an individual
% envelope
%% simulate response to stimuli envelopes
envelopes_path=sprintf(['C:/Users/ninet/Box/my box/LALOR LAB/', ...
    'oscillations project/MATLAB/Warped Speech/stimuli/wrinkle/', ...
    'regIrregEnvelopes128hz_%04dmsMax.mat'],config.irreg_maxt);
load(envelopes_path,"env");
% some cell entries intentionally left empty - only care about third row
% anyway
env=env(3,config.sim_trials);
switch config.normalize_envs
% normalize envelopes as in Doelling et al 2023
    case 'doelling'
        env=cellfun(@(x) norm_env_doelling(x),env,'UniformOutput',false);
    case 'range'
        % normalize to 0,1
        env=cellfun(@(x) normalize(x, 'range'),env,'UniformOutput',false);
end

%%
% run sl model for each trial's envelope OR demo k limit
if config.demo_klim
    % show deviation from limit cycle as stimulus exrets maximum force away
    % from limit cycle
    sl_responses=cell(1,2);
    [sl_responses{1,:}]=run_sl_env(config,sl_param,ones(size(env{1})));
else
    sl_responses=cell(length(env),2);
    for ii=1:length(env)
        fprintf('running %s model for %d/%d\n',config.model,ii,length(env))
        switch config.model
            case 'env'
                [sl_responses{ii,:}]=run_sl_env(config,sl_param,env{ii});
            case 'reset'
                [sl_responses{ii,:}]=run_sl_reset(config,sl_param);
            otherwise
                error('config.model: %s', config.model)
        end
    end
end
%% look at "phase portrait," xy, and input-output plots for a particular trial
if ~isempty(config.plot_individual_trials)
    for tt=1:length(config.plot_individual_trials)
        if isequal(config.sim_trials,1:120)
            plot_idx=config.plot_individual_trials(tt);
        else
            plot_idx=tt;
        end
        figure('Color', 'white')
        %note: time returned by model simulation should match fs samples of
        %stimulus because of how we defined tspan arg of ode45
        tstr_=sprintf('normalized stimulus envelope -trial %d, %04dms max', ...
            plot_idx,config.irreg_maxt);
        plot(sl_responses{plot_idx,1},env{plot_idx})
        xlabel('time (s)')
        ylabel('S(t)')
        title(tstr_)

        %xy plot
        tstr_=sprintf('%s speech response - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
        figure('Color', 'white')
        plot(sl_responses{plot_idx,1},[sl_responses{plot_idx,2}, env{plot_idx}])
        title(tstr_)
        legend('x', 'y','env')
        xlabel('time (s)')
        xlim([min(t) max(t)])
        hold off
        

        % "phase portrait"
        figure('Color', 'white')
        plot(sl_responses{plot_idx,2}(:,1),sl_responses{plot_idx,2}(:,2))
        hold on
        thetas_=0:pi/100:2*pi;
        plot(rs*cos(thetas_),rs*sin(thetas_),'r--')
        hold off
        clear thetas_
        axis equal
        xlabel('x')
        ylabel('y')
        legend('SL simulation','limit cycle')
       
        tstr_=sprintf('%s speech-phase portrait - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
        title(tstr_)

        
        % input-output (just x)
        % NOTE: if time in sl doesn't line up with stimulus samples this is
        % not gonna correspond right...
        figure('Color', 'white')
        plot(env{plot_idx},sl_responses{plot_idx,2}(:,1))
        xlabel('S(t)')
        ylabel('x(t)')
        tstr_=sprintf('%s input-output - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
        title(tstr_)
        clear tstr_
    end
end
%% derive TRFs from simulated responses
if config.fit_trfs
    trf_config=[];
    trf_config.add_noise=false;
    trf_config.optimize_lambda=false;
    trf_config.lam_range=10.^(-3:8);
    % note: used 400ms previously but if we're looking at effects for syllables
    % up to 1000ms apart... think it makes sense to extend the range a bit
    trf_config.cvtlims=[0, 400];
    trf_config.trtlims=[-500,800];
    % add 1/f noise to simulated responses to make them "realistic"
    % Oganian et al 2023 did this by using firls on gaussian white noise
    % they also used an snr of 1/10... not sure why
    %note: leaving this out for now to look at "pure" model response behavior 
    if trf_config.add_noise
        %TODO: add 1/f noise
    end
    
    % optimize lambda via cv
    % NOTE: although model responses derived from zero-mean, normalized
    % envelopes, we previously found TRF fitting benefits from normalizing by
    % std ONLY (i.e. keep them rectified...)
    
    %note: here we're scaling by each envelope's standard deviation, but we
    %previously used std(all envelopes) for simplicity...
    trf_env=cellfun(@(x) normalize(x,"scale"), env,'UniformOutput',false);
    trf_response=cellfun(@(x) x(:,1), sl_responses(:,2),'UniformOutput',false);
    if trf_config.optimize_lambda
        stats_obs=mTRFcrossval(trf_env,trf_response,config.fs,1, ...
            trf_config.cvtlims(1),trf_config.cvtlims(2),trf_config.lam_range);
        % plot TRFs
        % plot lambda tuning curve, select best lambda
        r_tuning_curve=mean(stats_obs.r, 1);
        [~,max_lam_idx]=max(r_tuning_curve);
        trf_config.best_lam=trf_config.lam_range(max_lam_idx);
        figure('Color', 'white')
        plot(trf_config.lam_range,r_tuning_curve)
        title('tuning curve')
        xlabel('\lambda')
        ylabel('r (crossvalidated)')
    else
        % seems unnecessary if no noise
        trf_config.best_lam=0;
    end
    %% train using best lambda on entire dataset
    trf_model=mTRFtrain(trf_env,trf_response,config.fs,1, ...
        trf_config.trtlims(1),trf_config.trtlims(2),trf_config.best_lam);
    %% plot model-TRF
    figure('Color', 'white'), plot(trf_model.t,trf_model.w)
    xticks(trf_config.trtlims(1):100:trf_config.trtlims(2))
    xlim([-100,trf_config.trtlims(2)])
    title(sprintf('%s-TRF %d max irreg interval',config.model,config.irreg_maxt));
end
%% quantify event-based phase concentration of output across all stimuli?
% to what end?
%


%% helpers
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
% [t, model_out]=run_sl_model(config,sl_param,input_fun)
% arguments
%     config (1,1) struct
%     sl_param (1,1) struct
%     s_data double = 0;
% end
if ~exist('s_data','var')
    tspan=0:1/config.fs:config.tmax_zero_input;
    input_fun=@(t,x) 0; % zero-function
else
    % envelope was given - convert to "continuous time" for ode solver via
    % anonymous function with interp
    tspan=0:1/config.fs:(length(s_data)-1)/config.fs;
    input_fun=@(t,~) interp1(tspan,s_data,t,"pchip",0);
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

% use ode solver to get sl model output:
[t, model_out]=ode45(@(t,x) sl_model(t,x,sl_param,input_fun),tspan,x0);
end

function dxdt=sl_model(t, x, sl_param, s_fun)
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

