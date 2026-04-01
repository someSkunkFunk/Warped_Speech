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