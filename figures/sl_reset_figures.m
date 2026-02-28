%% Stuart-Landau Oscillator with Hard Phase Reset + mTRF
% Equations (polar form):
%   r_dot = lambda*r - gamma*r^3
%   theta_dot = omega
%   At reset times t_k: theta(t_k) = theta_r

clear; clc; close all;

%% Parameters
stimulus_type = 'fast-slow';

sl = struct('param', [], 'result', []);
sl.param.f_nat   = 4;       % natural frequency in Hz (converted to rad in model)
sl.param.lambda  = 0.1;     % growth/decay rate
sl.param.gamma   = 13.83;   % nonlinear saturation
sl.param.theta_r = pi/2;    % preferred (reset) phase [rad]

%% TRF parameters
trf_param.tmin    = -200;   % ms
trf_param.tmax    =  800;   % ms
trf_param.lambda  = 0.1;    % mTRF regularisation (ridge)
trf_param.dir     = 1;      % 1 = forward (env -> x)

%% Load stimuli and extract event times
switch stimulus_type
    case 'fast-slow'
        [stim_env, event_times, trf_stim] = get_fast_slow_events();

        n_cond = size(event_times, 1);
        n_stim = size(event_times, 2);

        % Build sim_param struct array (one entry per condition)
        sim_param = struct('dt', [], 'T', [], 't', [], 'N', []);
        for cc = 1:n_cond
            cond_labels={'fast', 'original', 'slow'};
            sim_param(cc).dt = 1 / trf_stim.fs;
            sim_param(cc).T  = (length(trf_stim.envs{cc,1}) - 1) / trf_stim.fs;
            sim_param(cc).t  = 0 : sim_param(cc).dt : sim_param(cc).T;
            sim_param(cc).N  = length(sim_param(cc).t);
        end

    otherwise
        error('Invalid stimulus_type: %s', stimulus_type);
end

%% Simulate SL oscillator for every condition x stimulus
% x_all{cc,ss} = oscillator output vector for condition cc, stimulus ss
x_all = cell(n_cond, n_stim);

fprintf('Simulating SL oscillator (%d conditions x %d stimuli)...\n', n_cond, n_stim);
for cc = 1:n_cond
    for ss = 1:n_stim
        x_all{cc,ss} = sl_reset(sl, event_times{cc,ss}, sim_param(cc));
    end
    fprintf('  Condition %d done.\n', cc);
end

%% Limit-cycle amplitude
r_lc = sqrt(sl.param.lambda / sl.param.gamma);
fprintf('Limit-cycle amplitude r* = %.4f\n', r_lc);

%% ---- Visualisation: first stimulus of each condition, first 2 s ----
vis_dur = 2;   % seconds to show
cond_colors = lines(n_cond);

fig_vis = figure('Color', 'w', 'Position', [80 80 1000 220*n_cond]);

for cc = 1:n_cond
    fs  = trf_stim.fs;
    t_c = sim_param(cc).t;
    vis_mask = t_c <= vis_dur;
    t_vis    = t_c(vis_mask);

    % --- envelope (stimulus) ---
    ax_s = subplot(n_cond*2, 1, (cc-1)*2 + 1);
    env_vis = trf_stim.envs{cc,1};
    env_vis = env_vis(vis_mask);
    plot(t_vis, env_vis, 'Color', [0.4 0.4 0.4], 'LineWidth', 1);
    hold on;
    % mark event times within window
    ev = event_times{cc,1};
    ev = ev(ev <= vis_dur);
    stem(ev, max(env_vis)*ones(size(ev)), 'r', 'Marker', 'none', 'LineWidth', 0.8);
    ylabel('Env', 'FontSize', 10);
    title(sprintf('%s — envelope & events', cond_labels{cc}), 'FontSize', 11);
    set(ax_s, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out');
    xlim([0 vis_dur]);

    % --- oscillator output x(t) ---
    ax_x = subplot(n_cond*2, 1, (cc-1)*2 + 2);
    x_vis = x_all{cc,1}(vis_mask);
    plot(t_vis, x_vis, 'Color', cond_colors(cc,:), 'LineWidth', 1.4);
    hold on;
    stem(ev, max(abs(x_vis))*ones(size(ev)), 'r', 'Marker', 'none', 'LineWidth', 0.8);
    ylabel('x = r·cos(θ)', 'FontSize', 10);
    xlabel('Time (s)', 'FontSize', 10);
    set(ax_x, 'Box', 'off', 'TickDir', 'out');
    xlim([0 vis_dur]);

    linkaxes([ax_s ax_x], 'x');
end
sgtitle(sprintf('SL oscillator response — first 2 s (\\theta_r = %.2f rad)', sl.param.theta_r), ...
    'FontSize', 13, 'FontWeight', 'bold');

%% ---- mTRF: train one model per condition ----
% mTRF toolbox: mTRFtrain(stim, resp, fs, dir, tmin, tmax, lambda)
%   stim : [time x 1]  — envelope
%   resp : [time x 1]  — oscillator output x(t)
% We concatenate all stimuli within a condition before fitting.

fprintf('\nTraining mTRF models...\n');
trf_models = cell(n_cond, 1);
trf_stats  = cell(n_cond, 1);

for cc = 1:n_cond
    % Build cell arrays for mTRFcrossval / mTRFtrain
    % mTRF toolbox accepts cell arrays {stim1, stim2, ...} for multi-trial fitting
    stim_cell = cell(1, n_stim);
    resp_cell = cell(1, n_stim);

    for ss = 1:n_stim
        % Ensure column vectors and matching lengths
        env_ss = trf_stim.envs{cc,ss}(:);
        x_ss   = x_all{cc,ss}(:);

        % Trim to same length (rounding may cause ±1 sample difference)
        n_use = min(length(env_ss), length(x_ss));
        stim_cell{ss} = env_ss(1:n_use);
        resp_cell{ss} = x_ss(1:n_use);
    end

    % Train forward TRF (envelope predicts oscillator output)
    trf_models{cc} = mTRFtrain(stim_cell, resp_cell, trf_stim.fs, ...
        trf_param.dir, trf_param.tmin, trf_param.tmax, trf_param.lambda);

    fprintf('  Condition %d TRF trained.\n', cc);
end

%% ---- Plot TRF weights per condition ----
lags_ms = trf_models{1}.t;   % lag axis in ms (from mTRF model struct)

fig_trf = figure('Color', 'w', 'Position', [120 120 700 420]);
ax_trf  = axes('Parent', fig_trf);
hold(ax_trf, 'on');


for cc = 1:n_cond
    % weights: [lags x channels] — single channel here
    w = trf_models{cc}.w(:);   % column vector of weights
    plot(ax_trf, lags_ms, w, 'Color', cond_colors(cc,:), 'LineWidth', 2);
end

xline(ax_trf, 0, 'k--', 'LineWidth', 1);
yline(ax_trf, 0, 'k:', 'LineWidth', 0.8);
xlabel(ax_trf, 'Lag (ms)', 'FontSize', 12);
ylabel(ax_trf, 'TRF weight', 'FontSize', 12);
title(ax_trf, 'Forward TRF: envelope → x(t)', 'FontSize', 13, 'FontWeight', 'bold');
legend(ax_trf, cond_labels, 'Location', 'best', 'FontSize', 11);
set(ax_trf, 'Box', 'off', 'TickDir', 'out');
grid(ax_trf, 'on');
xlim(ax_trf, [trf_param.tmin trf_param.tmax]);

%% ---- Phase-response curve ----
n_grid  = 200;
thetas  = linspace(-pi, pi, n_grid);
phis    = sl.param.theta_r - thetas;

fig_prc = figure('Color', 'w', 'Position', [150 150 500 420]);
plot(thetas, phis, 'b', 'LineWidth', 1.8); hold on;

x_int = sl.param.theta_r;
plot([0, x_int], [0, 0], 'r--', 'LineWidth', 2);
plot(x_int, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(x_int, -0.15, '\theta_r', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top', 'FontSize', 12, 'Color', 'r');

refline(0, 0);
xlabel('\theta', 'FontSize', 12);
ylabel('\Delta\theta = \theta_r - \theta', 'FontSize', 12);
title('\theta-reset phase-response curve', 'FontSize', 13);
xlim([-pi pi]); ylim([-pi pi]);
grid on;

% Nice pi-fraction tick labels
x_ticks = -pi : pi/4 : pi;
set(gca, 'XTick', x_ticks, 'YTick', x_ticks, 'TickLabelInterpreter', 'latex', 'FontSize', 11);
tick_labels = make_pi_labels(x_ticks);
xticklabels(tick_labels);
yticklabels(tick_labels);


%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function x = sl_reset(sl, event_times, sim_param)
% Simulate Stuart-Landau oscillator with hard phase resets.
    omega = 2 * pi * sl.param.f_nat;

    dt = sim_param.dt;
    t  = sim_param.t;
    N  = sim_param.N;

    r0     = sqrt(sl.param.lambda / sl.param.gamma);
    theta0 = 0;

    r     = zeros(1, N);
    theta = zeros(1, N);
    r(1)     = r0;
    theta(1) = theta0;

    event_times = sort(event_times(:)');
    event_idx   = 1;

    f_r = @(ri) sl.param.lambda * ri - sl.param.gamma * ri^3;

    for i = 1:N-1
        % Hard phase reset if this sample coincides with an event
        if event_idx <= length(event_times) && ...
                abs(t(i) - event_times(event_idx)) < dt/2
            theta(i)  = sl.param.theta_r;
            event_idx = event_idx + 1;
        end

        % RK4 for r
        ri = r(i);
        k1 = f_r(ri);
        k2 = f_r(ri + dt/2 * k1);
        k3 = f_r(ri + dt/2 * k2);
        k4 = f_r(ri + dt   * k3);
        r(i+1)     = ri + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
        theta(i+1) = theta(i) + omega * dt;
    end

    % Final sample reset check
    if event_idx <= length(event_times) && ...
            abs(t(N) - event_times(event_idx)) < dt/2
        theta(N) = sl.param.theta_r;
    end

    x = r .* cos(theta);
end

% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function labels = make_pi_labels(ticks)
% Returns LaTeX-formatted cell array of pi-fraction labels for a tick vector.
    labels = cell(1, length(ticks));
    for k = 1:length(ticks)
        n = round(ticks(k) / (pi/4));   % numerator in units of pi/4
        if n == 0
            labels{k} = '$0$';
        elseif mod(n, 4) == 0
            c = n / 4;
            if     c ==  1, labels{k} = '$\pi$';
            elseif c == -1, labels{k} = '$-\pi$';
            else,           labels{k} = sprintf('$%d\\pi$', c);
            end
        elseif mod(n, 2) == 0
            c = n / 2;
            if     c ==  1, labels{k} = '$\frac{\pi}{2}$';
            elseif c == -1, labels{k} = '$-\frac{\pi}{2}$';
            else,           labels{k} = sprintf('$\\frac{%d\\pi}{2}$', c);
            end
        else
            if     n ==  1, labels{k} = '$\frac{\pi}{4}$';
            elseif n == -1, labels{k} = '$-\frac{\pi}{4}$';
            else,           labels{k} = sprintf('$\\frac{%d\\pi}{4}$', n);
            end
        end
    end
end
