% trf_plot_lib.m
% Shared plotting and analysis functions for TRF visualization scripts.
%
% Used by:
%   plot_trfs.m      - grand-average analysis
%   plot_ind_trfs.m  - individual-subject analysis
%
% MATLAB does not support true modules; place this file on the MATLAB path
% or in the same directory as the scripts that call it. Functions defined
% here are accessible as ordinary named functions.
%
% CONTENTS
%   Plotting
%     plot_butterfly_gfp        - stacked butterfly + GFP figure with component boundaries
%     plot_component_topos      - one topoplot figure per component window
%     plot_gfp                  - standalone single-condition GFP figure
%     legend_helper             - attach a color-keyed legend to a stacked plot axis
%     snr_plot                  - SNR vs subject-count plot
%
%   Data / computation
%     compute_gfp               - compute GFP (std across channels) for a model array
%     extract_component_stats   - measure peak latency + amplitude per component window
%     construct_avg_models      - average ind_models across subjects
%     estimate_snr              - RMS signal-to-noise ratio for a model array


% =========================================================================
%  PLOTTING
% =========================================================================

function h = plot_butterfly_gfp(model, gfp_row, active_result, ...
                                 cond_name, butterfly_fig, ...
                                 gfp_ylim, trf_ylim, baseline_val)
% PLOT_BUTTERFLY_GFP  Stacked butterfly + GFP axes with component boundaries.
%
%   h = plot_butterfly_gfp(model, gfp_row, t, active_result, cond_name,
%                          butterfly_fig, gfp_ylim, trf_ylim, baseline_val)
%
%   Inputs
%     model        - single mTRF model struct (w: [1 x time x chans])
%     gfp_row      - [1 x time] GFP vector for this condition
%     t            - [1 x time] time axis in ms
%     active_result- scalar struct with fields .starts, .ends, .topos
%                    (one row per component; ms times)
%     cond_name    - string, condition label (used for color lookup + title)
%     butterfly_fig- param struct (fields: t_lims, lw, condition_colors)
%     gfp_ylim     - [1x2] y-limits for GFP axis
%     trf_ylim     - [1x2] y-limits for butterfly axis
%     baseline_val - scalar; plotted as dashed threshold line on GFP axis
%
%   Output
%     h - struct with handles: .fig, .ax_gfp, .ax_trf, .butterfly_lines,
%                               .gfp_line, .baseline_line

t=model.t;

cond_color = butterfly_fig.condition_colors.(cond_name);
baseline_color = [.85 .85 .85];

h.fig = figure('Units','inches','Position',[0 0 7 3], ...
    'Name', sprintf('GFP with components - %s', cond_name), ...
    'Color','w');

% --- axis layout (normalized figure units) ---
ax_gfp_pos = [0.12 0.70 0.83 0.22];
ax_trf_pos  = [0.12 0.12 0.83 0.58];

% --- GFP axis ---
h.ax_gfp = axes('Position', ax_gfp_pos); 
h.gfp_line = plot(t, gfp_row, 'Color', cond_color);
title(cond_name)
grid on
hold(h.ax_gfp, 'on')
h.baseline_line = plot(h.ax_gfp, butterfly_fig.t_lims, ...
    [baseline_val, baseline_val], '--', 'Color', baseline_color, 'LineWidth', 1);
hold(h.ax_gfp, 'off')
ylabel('GFP')
set(h.ax_gfp, 'XTickLabel', [], 'YLim', gfp_ylim)
box off

% --- Butterfly axis ---
h.ax_trf = axes('Position', ax_trf_pos);
h.butterfly_lines = plot(t, squeeze(model.w));
grid on
set(h.butterfly_lines, 'LineWidth', butterfly_fig.lw, 'Color', cond_color)
set(h.ax_trf, 'YLim', trf_ylim)
ylabel('Amplitude')
xlabel('Time (ms)')
box off

% --- Component boundary annotations ---
n_components = length(active_result.starts);
for kk = 1:n_components
    t1 = active_result.starts(kk);
    t2 = active_result.ends(kk);
    T  = repmat([t1, t2], 2, 1);
    hold(h.ax_gfp, 'on');
    plot(h.ax_gfp, T, gfp_ylim, '--k')
    hold(h.ax_gfp, 'off');
    hold(h.ax_trf, 'on');
    plot(h.ax_trf, T, trf_ylim, '--k')
    hold(h.ax_trf, 'off');
end

% --- Synchronize x-axes ---
linkaxes([h.ax_gfp, h.ax_trf], 'x')
xlim(butterfly_fig.t_lims)

end


% -------------------------------------------------------------------------

function plot_component_topos(active_result, chanlocs, cond_name)
% PLOT_COMPONENT_TOPOS  One topoplot figure per component in active_result.
%
%   plot_component_topos(active_result, chanlocs, cond_name)
%
%   Inputs
%     active_result - scalar struct: .starts [n x 1], .ends [n x 1],
%                     .topos [n x n_chans]  (all in ms)
%     chanlocs      - EEGLAB chanlocs struct
%     cond_name     - string label for figure titles

n_components = size(active_result.topos, 1);
for kk = 1:n_components
    figure('Units','inches','Position',[0 0 1 1.2], ...
        'Name', sprintf('Component topo %s %0.0f-%0.0fms', ...
            cond_name, active_result.starts(kk), active_result.ends(kk)))
    topoplot(active_result.topos(kk,:), chanlocs)
    title(sprintf('%s, %0.0f\x2013%0.0f ms', cond_name, ...
        active_result.starts(kk), active_result.ends(kk)))
    ax = gca;
    ax.LooseInset = max(ax.TightInset, 0.02);
end

end


% -------------------------------------------------------------------------

function h = plot_gfp(gfp, model, cc, experiment_conditions, trf_fig_param)
% PLOT_GFP  Standalone GFP-only figure for condition cc.
%
%   h = plot_gfp(gfp, model, cc, experiment_conditions, trf_fig_param)
%
%   Inputs
%     gfp                  - [n_conditions x time] GFP matrix
%     model                - mTRF model struct for condition cc (used for .t)
%     cc                   - condition index
%     experiment_conditions- cell array of condition label strings
%     trf_fig_param        - param struct (fields: ylims, t_lims, condition_colors)
%
%   Output
%     h - struct with handles: .fig, .ax, .line, .title

cond_name = experiment_conditions{cc};
h.fig   = figure('Name', sprintf('GFP-only %s', cond_name), 'Color', 'White');
h.ax    = axes(h.fig);
hold(h.ax, 'on');
h.line  = plot(model.t, gfp(cc,:), ...
    'Color', trf_fig_param.condition_colors.(cond_name));
set(h.ax, 'YLim', trf_fig_param.ylims, 'XLim', trf_fig_param.t_lims)
h.title = title(h.ax, sprintf('GFP - %s', cond_name));

end


% -------------------------------------------------------------------------

function lh = legend_helper(ax, color_labels, color_rgbs)
% LEGEND_HELPER  Attach a correctly-colored legend to a stacked-condition axis.
%
%   lh = legend_helper(ax, color_labels, color_rgbs)
%
%   Inputs
%     ax           - axes handle containing the plotted lines
%     color_labels - cell array of condition name strings
%     color_rgbs   - struct mapping condition name -> [1x3] RGB

line_objs   = cell2struct(cell(size(color_labels))', color_labels);
legend_lines = zeros(1, numel(color_labels));
for ff = 1:numel(color_labels)
    lname = color_labels{ff};
    line_objs.(lname) = findobj(ax, 'Type', 'Line', 'Color', color_rgbs.(lname));
    legend_lines(ff)  = line_objs.(lname)(1);
end
lh = legend(legend_lines, color_labels);

end


% -------------------------------------------------------------------------

function snr_plot(snr_per_subj, conditions)
% SNR_PLOT  Plot SNR vs accumulated subject count for each condition.
%
%   snr_plot(snr_per_subj, conditions)
%
%   Inputs
%     snr_per_subj - [n_subjs x n_conditions] SNR matrix
%     conditions   - cell array of condition label strings

[n_subjs, n_cond] = size(snr_per_subj);
figure('Color','w','Name','SNR vs n_subjects')
hold on
for cc = 1:n_cond
    plot(1:n_subjs, snr_per_subj(:,cc))
end
hold off
legend(conditions)
xlabel('n subjects')
ylabel('SNR')
grid on

end


% =========================================================================
%  DATA / COMPUTATION
% =========================================================================

function gfp = compute_gfp(models, experiment_conditions)
% COMPUTE_GFP  Global field power (std across channels) for a model array.
%
%   gfp = compute_gfp(models, experiment_conditions)
%
%   Uses std (not RMS) because data are assumed to be mastoid-referenced.
%
%   Inputs
%     models               - [1 x n_conditions] mTRF model struct array;
%                            each entry has .w [1 x time x chans]
%     experiment_conditions- cell array of condition label strings
%
%   Output
%     gfp - [n_conditions x time] GFP matrix

n_conditions = numel(experiment_conditions);
n_time       = size(models(1).w, 2);
gfp          = nan(n_conditions, n_time);
for cc = 1:n_conditions
    W         = squeeze(models(cc).w);   % [time x chans]
    gfp(cc,:) = std(W, 0, 2);
end

end


% -------------------------------------------------------------------------

function stats = extract_component_stats(model, gfp_row, t, active_result)
% EXTRACT_COMPONENT_STATS  Measure TRF properties within grand-average component windows.
%
%   stats = extract_component_stats(model, gfp_row, t, active_result)
%
%   For each component window defined in active_result, returns:
%     - GFP peak latency (ms) within the window
%     - GFP peak amplitude within the window
%     - Mean GFP amplitude within the window
%
%   Inputs
%     model        - single mTRF model struct (w: [1 x time x chans])
%     gfp_row      - [1 x time] GFP for this model/condition
%     t            - [1 x time] time axis (ms)
%     active_result- scalar struct: .starts [n x 1], .ends [n x 1] (ms)
%
%   Output
%     stats - struct with fields (each [n_components x 1]):
%               .peak_latency_ms
%               .peak_amplitude
%               .mean_amplitude

n_components         = length(active_result.starts);
stats.peak_latency_ms = nan(n_components, 1);
stats.peak_amplitude  = nan(n_components, 1);
stats.mean_amplitude  = nan(n_components, 1);

for kk = 1:n_components
    win_m = t >= active_result.starts(kk) & t <= active_result.ends(kk);
    if ~any(win_m)
        warning('extract_component_stats: component %d window [%g %g] ms has no samples.', ...
            kk, active_result.starts(kk), active_result.ends(kk))
        continue
    end
    gfp_win = gfp_row(win_m);
    t_win   = t(win_m);
    [pk_amp, pk_idx]          = max(gfp_win);
    stats.peak_latency_ms(kk) = t_win(pk_idx);
    stats.peak_amplitude(kk)  = pk_amp;
    stats.mean_amplitude(kk)  = mean(gfp_win);
end

end


% -------------------------------------------------------------------------

function avg_model = construct_avg_models(ind_models)
% CONSTRUCT_AVG_MODELS  Average mTRF model weights across subjects.
%
%   avg_model = construct_avg_models(ind_models)
%
%   Inputs
%     ind_models - [n_subjs x n_conditions] mTRF model struct array
%
%   Output
%     avg_model  - [1 x n_conditions] averaged model struct
%
%   NOTE: only the .w field is meaningfully averaged; all other fields are
%   copied from subject 1. Fields that legitimately vary across subjects
%   (e.g. best_lam) should not be read from avg_model.

[n_subjs, n_conditions] = size(ind_models);
[~, n_weights, n_chans] = size(ind_models(1).w);

avg_model    = struct();
model_fields = fieldnames(ind_models(1,1));
warning(['construct_avg_models: only .w is correctly averaged. ', ...
    'Other fields are copied from subject 1 and may be inaccurate.'])

for cc = 1:n_conditions
    W_stack = nan(n_subjs, n_weights, n_chans);
    for ss = 1:n_subjs
        if ~isempty(ind_models(ss,cc).w)
            W_stack(ss,:,:) = ind_models(ss,cc).w;
        else
            warning('construct_avg_models: no weights for ss=%d, cc=%d', ss, cc)
        end
    end
    for ff = 1:numel(model_fields)
        ff_field = model_fields{ff};
        if strcmp(ff_field, 'w')
            if all(isnan(W_stack(:)))
                avg_model(1,cc).(ff_field) = [];
            else
                avg_model(1,cc).(ff_field) = mean(W_stack, 1, 'omitnan');
            end
        elseif strcmp(ff_field, 'b')
            continue   % bias: safe to skip
        else
            avg_model(1,cc).(ff_field) = ind_models(1,cc).(ff_field);
        end
    end
    avg_model(1,cc).avg = true;   % flag to distinguish from native toolbox model
end

end


% -------------------------------------------------------------------------

function snr = estimate_snr(avg_models, noise_window, signal_window)
% ESTIMATE_SNR  RMS-based SNR estimate for each condition in avg_models.
%
%   snr = estimate_snr(avg_models, noise_window, signal_window)
%
%   Inputs
%     avg_models    - [1 x n_conditions] mTRF model struct array
%     noise_window  - [1x2] time range (ms) for noise floor  (default [-200 0])
%     signal_window - [1x2] time range (ms) for signal window (default [100 300])
%
%   Output
%     snr - [1 x n_conditions] SNR values

arguments
    avg_models    (1,:) struct
    noise_window  (1,2) = [-200,   0]
    signal_window (1,2) = [ 100, 300]
end

n_conditions = size(avg_models, 2);
snr          = nan(1, n_conditions);

for cc = 1:n_conditions
    t            = avg_models(1,cc).t;
    noise_mask   = t > min(noise_window)  & t < max(noise_window);
    signal_mask  = t > min(signal_window) & t < max(signal_window);
    snr(cc)      = rms(avg_models(1,cc).w(signal_mask)) / ...
                   rms(avg_models(1,cc).w(noise_mask));
end

end