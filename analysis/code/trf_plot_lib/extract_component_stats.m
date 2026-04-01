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
%  DATA / COMPUTATION
% =========================================================================

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