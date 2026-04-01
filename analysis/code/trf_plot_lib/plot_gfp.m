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