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