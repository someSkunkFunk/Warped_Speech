
%% Stuart-Landau Oscillator with Hard Phase Reset
% Equations (polar form):
%   r_dot = lambda*r - gamma*r^3
%   theta_dot = omega
%   At reset times t_k: theta(t_k) = theta_r

clear; clc; close all;

%% Parameters
stimulus_type='fast-slow';
% hard-reset model settings
sl=struct('param',[],'result',[]);
sl.param.f_nat  = 4;        % natural frequency in Hz (converted to rad in model)
sl.param.lambda = 0.1;      % growth/decay rate
sl.param.gamma  = 13.83;    % nonlinear saturation

theta_r = pi/4;                        % preferred (reset) phase [rad]
sl.param.theta_r=theta_r;
%% Simulation time
switch stimulus_type
    case 'ex'
        sim_param.dt= 0.001;           % time step (s)
        sim_param.T= 1;               % total duration (s)
        sim_param.t= 0:sim_param.dt:sim_param.T;
        sim_param.N= length(sim_param.t);
        [stim, event_times]=get_rand_events(sim_param);
    case 'fast-slow'
        [stim, event_times,trf_stim]=get_fast_slow_events();
        % extract simulation params for each condition
        sim_param=struct('dt',[],'T',[],'t',[],'N',[]);
        for cc=1:size(event_times,1)
            sim_param(cc).dt=1/trf_stim.fs;
            sim_param(cc).T=(length(trf_stim.env{cc,1})-1)/trf_stim.fs;
            sim_param(cc).t=0:sim_param(cc).dt:sim_param(cc).T;
            sim_param(cc).N=length(sim_param(cc).t)
        end
        % warning('extract sim_param from stimuli! as (3 x 120 struct');
    case 'reg-irreg'
    otherwise
        error('invalid stimulus type: %s',stimulus_type)
end


%% simulate sl reset
x=sl_reset(sl, event_times, sim_param);

%% Compute limit-cycle amplitude for reference
r_lc = sqrt(sl.param.lambda / sl.param.gamma);
fprintf('Limit-cycle amplitude r* = %.4f\n', r_lc);

%% Plot
fig = figure('Color','w','Position',[100 100 900 500]);

% --- Stimulus ---
ax1 = subplot(2,1,1);
stem(sim_param.t(stim==1), ones(1,sum(stim)), 'k', 'Marker','none','LineWidth',1.2);
xlim([0 sim_param.T]);
ylim([0 1.4]);
ylabel('S(t)','FontSize',12);
title_string=sprintf('Stuart-Landau Oscillator with Hard Phase Reset (\\theta_r = %0.2f rad)',theta_r);
title(title_string,'FontSize',13,'FontWeight','bold');
set(ax1,'XTickLabel',[],'Box','off','TickDir','out');
grid on;

% --- x = r*cos(theta) ---
ax2 = subplot(2,1,2);
plot(sim_param.t, x, 'Color',[0.15 0.45 0.8], 'LineWidth',1.4);
hold on;
% Mark reset events on x trace
for k = 1:length(event_times)
    [~, idx] = min(abs(sim_param.t - event_times(k)));
    plot(sim_param.t(idx), x(idx), 'rv', 'MarkerFaceColor','r','MarkerSize',6);
end
xlim([0 sim_param.T]);
xlabel('Time (s)','FontSize',12);
ylabel('x = r\cdotcos(\theta)','FontSize',12);
legend('x(t)','Phase reset','Location','northeast','FontSize',10);
set(ax2,'Box','off','TickDir','out');
grid on;

linkaxes([ax1 ax2],'x');


%% -- Phase response curve -- 
% should actually be equivalent for any oscillatory
% configuration of the model (by definition)
n_grid=100;

thetas=linspace(-pi,2*pi, n_grid);
phis=theta_r-thetas;
% phis=linspace(-pi,pi, n_grid);

figure('Color','w'), plot(thetas,phis), hold on


% Define the range (adjust start/end as needed)
x_ticks = -pi: pi/4 : pi;

% Generate corresponding LaTeX tick labels
tick_labels = {};
for k = 1:length(x_ticks)
    n = round(x_ticks(k) / (pi/4));  % numerator multiplier (in units of pi/4)
    
    if n == 0
        tick_labels{end+1} = '$0$';
    elseif mod(n, 4) == 0
        % Whole multiple of pi
        coeff = n / 4;
        if coeff == 1
            tick_labels{end+1} = '$\pi$';
        elseif coeff == -1
            tick_labels{end+1} = '$-\pi$';
        else
            tick_labels{end+1} = sprintf('$%d\\pi$', coeff);
        end
    elseif mod(n, 2) == 0
        % Multiple of pi/2
        coeff = n / 2;
        if coeff == 1
            tick_labels{end+1} = '$\frac{\pi}{2}$';
        elseif coeff == -1
            tick_labels{end+1} = '$-\frac{\pi}{2}$';
        else
            tick_labels{end+1} = sprintf('$\\frac{%d\\pi}{2}$', coeff);
        end
    else
        % Odd multiple of pi/4
        if n == 1
            tick_labels{end+1} = '$\frac{\pi}{4}$';
        elseif n == -1
            tick_labels{end+1} = '$-\frac{\pi}{4}$';
        else
            tick_labels{end+1} = sprintf('$\\frac{%d\\pi}{4}$', n);
        end
    end
end
% x-intercept point
x_int = theta_r;
y_int = 0;

% Red dashed line from origin (0,0) to x-intercept (theta_r, 0)
plot([0, x_int], [0, y_int], 'r--', 'LineWidth', 2);

% Mark the x-intercept point
plot(x_int, y_int, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% Label the x-intercept on the x-axis
text(x_int, -0.1, '\theta_r', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', ...
    'FontSize', 12, ...
    'Color', 'r');

% Formatting
axisline = refline([0 0]); axisline.Color = 'k';
xlabel('\theta')
ylabel('\Delta\theta=\theta_r - \theta')

xlim([-pi pi])
ylim([-pi pi])
title('\theta-reset phase-response curve')
grid on;
hold off;
% Apply to axes
xticks(x_ticks);
xticklabels(tick_labels);
yticks(x_ticks)
yticklabels(tick_labels)
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize=11;


%%
function [stim,event_times]=get_rand_events(sim_param)
%% Random event times (stimulus)
% Example: periodic events at 4 Hz (matching natural frequency)
% dt=sim_param.dt;
% T=sim_param.T;
t=sim_param.t;
N=sim_param.N;
n_events=5;
event_times=[sort(rand(n_events,1))];

% Build binary stimulus signal for plotting
stim = zeros(1, N);
for k = 1:length(event_times)
    [~, idx] = min(abs(t - event_times(k)));
    stim(idx) = 1;
end

end
function x=sl_reset(sl,event_times,sim_param)
omega = 2 * pi * sl.param.f_nat;   % convert to rad/s

dt=sim_param.dt;           % time step (s)
t=sim_param.t;
N=sim_param.N;
%% Initial conditions
r0     = sqrt(sl.param.lambda / sl.param.gamma);  % start on limit cycle
theta0 = 0;

%% Integrate with phase resets
r     = zeros(1, N);
theta = zeros(1, N);
r(1)     = r0;
theta(1) = theta0;

% Pre-sort event times
event_times = sort(event_times);
event_idx = 1;   % pointer into event_times

for i = 1:N-1
    % Check if current time is an event time (within half a dt)
    if event_idx <= length(event_times) && ...
            abs(t(i) - event_times(event_idx)) < dt/2
        % Hard phase reset
        theta(i) = sl.param.theta_r;
        event_idx = event_idx + 1;
    end

    % RK4 integration for r (theta is simple, integrate analytically)
    r_i = r(i);
    f = @(ri) sl.param.lambda * ri - sl.param.gamma * ri^3;

    k1 = f(r_i);
    k2 = f(r_i + dt/2 * k1);
    k3 = f(r_i + dt/2 * k2);
    k4 = f(r_i + dt   * k3);

    r(i+1)     = r_i + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    theta(i+1) = theta(i) + omega * dt;
end

% Handle reset at last sample if needed
if event_idx <= length(event_times) && ...
        abs(t(N) - event_times(event_idx)) < dt/2
    theta(N) = sl_param.theta_r;
end

%% Compute Cartesian output
x = r .* cos(theta);   % x = r*cos(theta)

end

function [stim, event_times, trf_stim]=get_fast_slow_events()
    % phase-reset model doesn't care about amplitudes so ignoring them
    global boxdir_mine
    % NOTE: we didn't save updated fast-slow using oganian's full pipeline
    envs_path=fullfile(boxdir_mine,'stimuli','wrinkle','fastSlowEnvelopes128hz.mat');
    load(envs_path,'env','fs')
    trf_stim.envs=env;
    trf_stim.fs=fs;
    stim=cell(size(env));
    event_times=cell(size(env));
    for cc=1:size(env,1)
        for ss=1:size(env,2)
            temp_env_derivative=[0; diff(env{cc,ss})];
            [~,event_times{cc,ss}]=findpeaks(temp_env_derivative,fs);
        end
        
    end

end
