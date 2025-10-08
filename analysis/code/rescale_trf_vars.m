function [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
    trf_config)
% [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg,trf_config)
    resp=preprocessed_eeg.resp;
    if trf_config.zscore_envs
        % NOTE: need to check if they've already been z-scored before doing
        % this.... or not because it will always load from the mat file?
        if trf_config.norm_envs
            error('dont do both normalization and z-scoring on envelopes')
        end
        disp('z-scoring envelopes')
        load(trf_config.paths.envelopesFile,'mu','sigma');

        stim=cellfun(@(x) (x-mu)/sigma, stim,'UniformOutput',false);
    end
    if trf_config.norm_envs
        disp('normalizing envelopes')
        load(trf_config.paths.envelopesFile,'sigma');
        stim=cellfun(@(x) x/sigma, stim,'UniformOutput',false);
    end

    if trf_config.zscore_eeg
        disp('z-scoring eeg')
            % concatenate all trials
        resp_cat=cat(1,preprocessed_eeg.resp{:,:});
        % z-score all the channels together
        eeg_mu=mean(resp_cat,'all');
        eeg_sigma=std(resp_cat,0,'all');
        preprocessed_eeg.resp=cellfun(@(x)(x-eeg_mu)./eeg_sigma,resp,'UniformOutput',false);

    end
end
