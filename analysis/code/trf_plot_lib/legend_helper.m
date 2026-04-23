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
lh = legend(legend_lines, color_labels,'Location','southeast');

end


