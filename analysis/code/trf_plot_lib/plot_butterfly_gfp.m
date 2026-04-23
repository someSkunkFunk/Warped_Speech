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

h.fig = figure('Units','inches','Position',[1 1 6 2], ...
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


