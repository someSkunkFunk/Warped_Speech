% model AFO-WC response to speech
config=[];
config.init='limit cycle'; %limit cycle or rand (uniform [-1,1])
config.fs=128;
config.tmax_zero_input=60; % time limit in seconds for undriven input simulation
config.irreg_maxt=1000; % choose 1000, 750, or 500 ms max interval for irreg
config.plot_individual_trials=false;

%% select SL parameters
% optimal parameters used by wei ching

wc_param=[];
wc_param.a=10;
wc_param.b=10;
wc_param.c=10;
wc_param.g=-3;
wc_param.tau=1/17;
wc_pram.rhoI=-7;
wc_param.k=0.2;

wc_param.sil_tol=; % some value between [0,1] representing normalized stimulus values 
%% UNFORCED MODEL CHARACTERIZATION
%% run model without input
[t, wc_nostim]=run_wc_model(config,wc_param);
%% plot unforced model output
figure
plot(t,wc_nostim(:,1)), hold on
plot(t,wc_nostim(:,2))
title(sprintf('%s without stimulus',config.model))
legend('x', 'y')
xlabel('time (s)')
xlim([min(t) max(t)])
hold off
%% phase-portrait of unforced model output
figure
plot(wc_nostim(:,1),wc_nostim(:,2))
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
env=env(3,:);
% rectify the envelopes
env=cellfun(@(x) max(x,0), env);
% normalize envelopes as in Doelling et al 2023
nenv=cellfun(@(x) norm_env(x),env,'UniformOutput',false);
%%
% run sl model for each trial's envelope 
%TODO: how to enter envelope in function form below?
sl_responses=cell(length(nenv),2);
for ii=1:length(nenv)
    fprintf('running %s model for %d/%d\n',config.model,ii,length(nenv))
    [sl_responses{ii,:}]=run_wc_model(config,wc_param,nenv{ii});
end
%% look at "phase portrait," xy, and input-output plots for a particular trial
if config.plot_individual_trials
    for plot_idx=1:length(nenv)
        % plot_idx=25; % should probably look at all of them eventually
        %xy plot
        tstr_=sprintf('%s speech response - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
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
        tstr_=sprintf('%s speech-phase portrait - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
        title(tstr_)
        
        % input-output (just x)
        figure
        plot(nenv{plot_idx},sl_responses{plot_idx,2}(:,1))
        xlabel('S(t)')
        ylabel('x(t)')
        tstr_=sprintf('%s input-output - trial %d, %04dms max', ...
            config.model,plot_idx,config.irreg_maxt);
        title(tstr_)
        clear tstr_
    end
end
%% derive TRFs from simulated responses
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
    figure
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
figure, plot(trf_model.t,trf_model.w)
xticks(trf_config.trtlims(1):100:trf_config.trtlims(2))
xlim([-100,trf_config.trtlims(2)])
title(sprintf('%s-TRF %d max irreg interval',config.model,config.irreg_maxt));

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
function [t, model_out]=run_wc_model(config,wc_param,s_data)
% [t, model_out]=run_sl_model(config,sl_param,input_fun)
% arguments
%     config (1,1) struct
%     sl_param (1,1) struct
%     s_data double = 0;
% end
if ~exist('s_data','var')
    tspan=0:1/config.fs:config.tmax_zero_input;
    % input_fun=@(t,~) 0; % zero-function
    % define 0
else
    % envelope was given - convert to "continuous time" for ode solver via
    % anonymous function with interp
    tspan=0:1/config.fs:(length(s_data)-1)/config.fs;
end
% also needs "continous time" peakRate function & binary "silence or
% not" label based on config.sil_tol (which will just be some arbitrary
% global threshold for now....)
% input_fun=@(t,~) interp1(tspan,s_data,t,"linear",0);

function Sfeats=input_fun(t,tspan,s_data,wc_params.sil_tol)
% returns continuous time "interpolation" of envelope, peakRate, & silence
end

%TODO: how to flexibly switch between sl and wc models?
switch config.init
    case 'rand'
        x0=init_rand(3); 
    case 'limit cycle'
        % select random values already at the limit cycle
        warning('init_limit_cycle should be different here.')
        x0=init_limit_cycle(wc_param);
    otherwise
        error('initialization config param not recognized.')
end

% use ode solver to get sl model output:
[t, model_out]=ode45(@(t,x) wc_model(t,x,wc_param,input_fun),tspan,x0);
end

function dxdt=wc_model(t, x, wc_param, s_fun)
% dxdt=sl_model(t, x, sl_param, Sfun)
% t: times to sample SL activity at
% x: col vector of E, I, rhoE
% s_fun: stimulus input specified as a function 
% uses sigmoid function defined in Doelling et al 2019

%TODO: define some default values
%unpack parameters
% synaptic coupling
a=wc_param.a;
b=wc_param.b;
c=wc_param.c;
g=wc_param.g;
% stable inputs from distant brain areas
rhoI=wc_param.rhoI;
% membrane time const
tau=wc_param.tau; 
k=wc_param.k;
% sigmoid used in Doelling et al 2019 
sigmoid=@(z) 1/(1+exp(-z)); 
S=s_fun(t,x);
%TODO: estimate mean t more realistically/online
mean_t=;

if silence(S)
    %TODO: when does the brain know there is silence?
    rho0=-3.4;
else
    % during sound
    rho0=(0.45*mean_t/(mean_t-0.21))-3.6;
end
dxdt=nan(3,1);
% E differential equation
dxdt(1)=(1/tau).*(-x(1)+sigmoid(x(3)+c*x(1)-a*x(2)+k*S));
% I differential equation
dxdt(2)=(1/tau).*(-x(2)+sigmoid(rhoI+b*x(1)-g*x(2)));
% rhoE differential equation
dxdt(3)=-.045*(x(3)-rho0);
end
% plotting 
% function h=plot_xy()
% end

%references
% Doelling, Keith B., M. Florencia Assaneo, Dana Bevilacqua, Bijan Pesaran,
% and David Poeppel. “An Oscillator Model Better Predicts Cortical 
% Entrainment to Music.” Proceedings of the National Academy of Sciences 116,
% no. 20 (2019): 10113–21. https://doi.org/10.1073/pnas.1816414116.
%
% Doelling, Keith B., Luc H. Arnal, and M. Florencia Assaneo. 
% “Adaptive Oscillators Support Bayesian Prediction in Temporal 
% Processing.” PLOS Computational Biology 19, no. 11 (2023): e1011669. 
% https://doi.org/10.1371/journal.pcbi.1011669.

% Oganian, Yulia, Katsuaki Kojima, Assaf Breska, et al. 
% “Phase Alignment of Low-Frequency Neural Activity to the Amplitude 
% Envelope of Speech Reflects Evoked Responses to Acoustic Edges, 
% Not Oscillatory Entrainment.” The Journal of Neuroscience 43, 
% no. 21 (2023): 3909–21. https://doi.org/10.1523/JNEUROSCI.1663-22.2023.

