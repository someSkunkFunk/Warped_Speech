% trf_peak_latency
% identify peak in ind_model weights for each subject electrode at
% component_windows


peaktimes_subjlvl=cellfun(@(x) repmat({nan(n_subjs,128)},1,size(x,1)), ...
    component_windows,'UniformOutput',false);

for ss=1:n_subjs
    for cc=1:numel(experiment_conditions)
        w_=ind_models(ss,cc).w;
        t_=ind_models(ss,cc).t;
        for ci = size(component_windows{cc},1)
            win_=component_windows{cc}(ci,:); % indices, not time
      
            for ee=1:128
                [~,peak_]=findpeaks(w_(1,win_(1):win_(2),ee), ...
                    ind_models(ss,cc).fs,'NPeaks',1);
                if ~isempty(peak_)
                    peaktimes_subjlvl{cc}{ci}(ss,ee)=peak_;
                end
            end
        end
    end
end
%% --- scatterplot ---

%% --- do some statistical test across subjects ---
% should compare across conditions, within subjects



