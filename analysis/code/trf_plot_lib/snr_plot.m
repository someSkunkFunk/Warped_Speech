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
