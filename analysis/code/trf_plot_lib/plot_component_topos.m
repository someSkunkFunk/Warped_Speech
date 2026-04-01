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


