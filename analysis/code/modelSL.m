% model SL response
config=[];
config.init='limit cycle'; %limit cycle or rand (uniform [-1,1])
config.fs=128;
config.tmax_zero_input=60; % time limit in seconds for undriven input simulation
config.irreg_maxt=500; % choose 1000, 750, or 500 ms max interval for irreg
config.plot_individual_trials=false;
%% select SL parameters
% optimal parameters used by wei ching

sl_param=[];
sl_param.lambda=0.1;
sl_param.gamma=13.83;
sl_param.k=80;
sl_param.f_nat=4; % in Hz -> converted to radians when running model

%% UNFORCED MODEL CHARACTERIZATION
%% run model without input
[t, sl_nostim]=run_sl_model(config,sl_param);
%% plot unforced model output
figure
plot(t,sl_nostim(:,1)), hold on
plot(t,sl_nostim(:,2))
title('Stuart Landau without stimulus')
legend('x', 'y')
xlabel('time (s)')
xlim([min(t) max(t)])
hold off
%% phase-portrait of unforced model output
figure
plot(sl_nostim(:,1),sl_nostim(:,2))
axis equal
xlabel('x')
ylabel('y')
title('Undriven SL phase portrait')
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
env=env(3,:);
% normalize envelopes as in Doelling et al 2023
nenv=cellfun(@(x) norm_env(x),env,'UniformOutput',false);
%%
% run sl model for each trial's envelope 
%TODO: how to enter envelope in function form below?
sl_responses=cell(length(nenv),2);
for ii=1:length(nenv)
    fprintf('running sl model for %d/%d\n',ii,length(nenv))
    [sl_responses{ii,:}]=run_sl_model(config,sl_param,nenv{ii});
end
%% look at "phase portrait," xy, and input-output plots for a particular trial
if config.plot_individual_trials
    for plot_idx=1:length(nenv)
        % plot_idx=25; % should probably look at all of them eventually
        %xy plot
        tstr_=sprintf('Stuart Landau speech response - trial %d, %04dms max',plot_idx,config.irreg_maxt);
        figure
        plot(sl_responses{plot_idx,1},sl_responses{plot_idx,2})
        title(tstr_)
        legend('x', 'y')
        xlabel('time (s)')
        xlim([min(t) max(t)])
        hold off
        % "phase portrait"
        figure
        plot(sl_responses{plot_idx,2}(:,1),sl_responses{plot_idx,2}(:,2))
        axis equal
        xlabel('x')
        ylabel('y')
        tstr_=sprintf('SL speech-phase portrait - trial %d, %04dms max',plot_idx,config.irreg_maxt);
        title(tstr_)
        
        % input-output (just x)
        figure
        plot(nenv{plot_idx},sl_responses{plot_idx,2}(:,1))
        xlabel('S(t)')
        ylabel('x(t)')
        tstr_=sprintf('SL input-output - trial %d, %04dms max',plot_idx,config.irreg_maxt);
        title(tstr_)
        clear tstr_
    end
end
%% derive TRFs from simulated responses
trf_config=[];
trf_config.add_noise=false;
trf_config.lam_range=10.^(-3:8);
% note: used 400ms previously but if we're looking at effects for syllables
% up to 1000ms apart... think it makes sense to extend the range a bit
trf_config.tlims=[0, 400];
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
%%
stats_obs=mTRFcrossval(trf_env,trf_response,config.fs,1, ...
    trf_config.tlims(1),trf_config.tlims(2),trf_config.lam_range);

%% plot TRFs
% plot lambda tuning curve, select best lambda
r_tuning_curve=mean(stats_obs.r,1);
[~,max_lam_idx]=max(r_tuning_curve);
best_lam=trf_config.lam_range(max_lam_idx);
figure
plot(trf_config.lam_range,r_tuning_curve)
title('tuning curve')
xlabel('\lambda')
ylabel('r (crossvalidated)')
% train using best lambda on entire dataset


%% quantify event-based phase concentration of output across all stimuli


%% helpers
function env_normed=norm_env(env_data)
% normalize stim as in Doelling et al 2023
env_bar=mean(env_data);
env_normed=(env_data-env_bar)/max(env_data-env_bar);
end

% function s_out=env_fun(t,s_data,fs)
% % helper function for getting "continous time" stim envelope representation
% % to be used in ode solver
% t_data=0:1/fs:(length(s_data)-1)/fs;
% s_out=interp1(t_data,s_data,t,'linear',0);
% end

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
function [t, model_out]=run_sl_model(config,sl_param,s_data)
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
    input_fun=@(t,~) interp1(tspan,s_data,t,"linear",0);
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
dxdt(2)=lambda*x(2)+omega*x(1);
end
% plotting 
% function h=plot_xy()
% end

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

