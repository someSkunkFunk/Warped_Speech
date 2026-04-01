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