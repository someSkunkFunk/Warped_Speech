% model SL response
config=[];
config.init='limit cycle'; %limit cycle or rand (uniform [-1,1])
config.fs=128;
%% select SL parameters
% optimal parameters used by wei ching

sl_param=[];
sl_param.lambda=0.1;
sl_param.gamma=13.83;
sl_param.k=80;
sl_param.f_nat=4; % in Hz -> converted to radians when running model

%% UNFORCED MODEL CHARACTERIZATION
%% run model without input
zero_fun=@(t,x) 0;
[t, sl_out]=run_sl_model(config,sl_param,zero_fun);
%% plot unforced model output
figure
plot(t,sl_out(:,1)), hold on
plot(t,sl_out(:,2))
title('Stuart Landau without stimulus')
legend('x', 'y')
xlabel('time (s)')
hold off
%% phase-portrait of unforced model output
plot(sl_out(:,1),sl_out(:,2))
axis equal
xlabel('x')
ylabel('y')
title('Undriven SL phase portrait')
%% arnold tongue
%% FORCED RESPONSE CHARACTERIZATION
% goal: characterize model response to speech generally, not an individual
% envelope
%% simulate response to stimuli envelopes
envelopes_path="C:\Users\ninet\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\regIrregEnvelopes128hz_1000msMax.mat";
load(envelopes_path,"env");
% some cell entries intentionally left empty - only care about third row
% anyway
env=env(3,:);
nenv=cellfun(@(x) norm_env(x),env,'UniformOutput',false);
%%



%% quantify phase concentration of output

%% derive TRFs from simulated responses

%% helpers
function env_normed=norm_env(env_data)
% normalize stim as in Doelling et al 2023
env_bar=mean(env_data);
env_normed=(env_data-env_bar)/max(env_data-env_bar);
end

function s_out=env_fun(t,s_data,fs)
% helper function for getting "continous time" stim envelope representation
% to be used in ode solver
t_data=0:1/fs:(length(s_data)-1)/fs;
s_out=interp1(t_data,s_data,t,'linear',0);
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
function [t, model_out]=run_sl_model(config,sl_param,input_fun)
% [t, model_out]=run_sl_model(config,sl_param,input_fun)
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

tspan=0:1/config.fs:64;
% use ode solver to get sl model output:
[t, model_out]=ode45(@(t,x) sl_model(t,x,sl_param,input_fun),tspan,x0);
end

function dxdt=sl_model(t, x, sl_param, Sfun)
% dxdt=sl_model(t, x, sl_param, Sfun)
% t: times to sample SL activity at
% x: col vector of x and y in SL equations
% sl_param.lambda: oscillatory "switch" parameter
% sl_param.f_nat: natural frequency in Hz (converted to rads/s here)
% sl_param.gamma: nonlinear gain control
% sl_param.k: coupling parameter
% Sfun: stimulus input specified as a function

%TODO: define some default values
%unpack sl parameters
lambda=sl_param.lambda;
omega=sl_param.f_nat*2*pi;
gamma=sl_param.gamma;
k=sl_param.k;

r2=x(1)^2+x(2)^2;
S=Sfun(t,x);

dxdt=nan(2,1);
% x differential equation
dxdt(1)=lambda*x(1)-omega*x(2)-gamma*r2*x(1)+k*S;
% y differential equation
dxdt(2)=lambda*x(2)+omega*x(1);
end
%references
% Doelling, Keith B., Luc H. Arnal, and M. Florencia Assaneo. 
% “Adaptive Oscillators Support Bayesian Prediction in Temporal 
% Processing.” PLOS Computational Biology 19, no. 11 (2023): e1011669. 
% https://doi.org/10.1371/journal.pcbi.1011669.
