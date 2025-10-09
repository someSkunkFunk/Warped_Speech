    
%% config analysis
clearvars -except user_profile boxdir_mine boxdir_lab
clc

%TODO 2: clean up trf and preprocess config param setting (check that it
%works with fast/slow)

%what's left: code now shoud run pretty seamlessly except for clunkiness in
%loading best lam when doing separate conditions... also realizing that
%having only one set of tmin,tmax values in trf_config has yoked the model
%train time params to be yoked to the crossvalidate train params if we want
%them to be saved to same file

% TODO 3: extend code functionality to reg/irreg
% TODO 3: look at reg trf (does it exist??)
% for subj=[2:7,9:22]
for subj=[2]
clearvars -except user_profile boxdir_mine boxdir_lab subj
close all
%% setup configs
preprocess_config.subj=subj;
trf_config.subj=subj;
trf_config.separate_conditions=false;
do_nulltest=true;

if ~trf_config.separate_conditions
    trf_config.do_lambda_optimization=true;
    % do full crossvalidation only to get optimal lambda in
    % condition-agnostic way
    trf_config.separate_conditions=false;
    % we need to crossvalidate to optimize lambds
    trf_config.crossvalidate=true;
    % in this case, we only use causal trf window that is shorter to speed
    % up process
    trf_config.tmin_ms=0;
    trf_config.tmin_ms=400;
else
    % we assume optimization done data from all conditions and get best
    % lambda from saved checkpoint
    trf_config_=trf_config;
    trf_config_.separate_conditions=false;
    trf_config_.do_lambda_optimization=true;
    
    trf_config.do_lambda_optimization=false;
end
% if doing nulltest, we also need stats_obs from cross validate regardless
% of conditions separate or not
if do_nulltest && ~trf_config.crossvalidate
    trf_config.crossvalidate=true;
end

preprocess_config=config_preprocess(preprocess_config);
trf_config=config_trf(trf_config,preprocess_config);




%% check if preprocessed data exists...
pp_checkpoint_=load_checkpoint(preprocess_config);
% if ~isempty(preprocessed_eeg)
%     stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
% % if exist(preprocess_config.preprocessed_eeg_path,'file') && ...
% %     configs_match(preprocess_config.preprocessed_eeg_path,preprocess_config)
% %     fprintf(['existing new-fmt preprocessed mat file found, loading ' ...
% %         'from %s.\n'],preprocess_config.preprocessed_eeg_path)
% %     preprocess_checkpoint=...
% %         load_checkpoint(preprocess_config.preprocessed_eeg_path,preprocess_config);
% 
%     % if preprocess_checkpoint.desired_config_found
%     % preload_preprocessed=true;
%     % preprocess_config=preprocess_checkpoint.preprocess_config;
%     % preprocessed_eeg=preprocess_checkpoint.preprocessed_eeg;
%     % clear preprocess_checkpoint
%     % stim=load_stim_cell(preprocess_config,preprocessed_eeg);
% 
% end

%% preprocess from raw (bdf)
if isempty(pp_checkpoint_)
    fprintf('processing from bdf...\n')
    preprocessed_eeg=preprocess_eeg(preprocess_config);
    stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
    % trim resp to have same durations as stim (only need to do during
    % preprocessing)
    preprocessed_eeg=remove_excess_epoch(stim,preprocessed_eeg);
    fprintf('saving to %s\n',preprocess_config.paths.output_dir)
    save_checkpoint(preprocessed_eeg,preprocess_config)
    preload_preprocessed=false;
else
    preprocessed_eeg=pp_checkpoint_.preprocessed_eeg;
    stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
    preload_preprocessed=true;
end
clear pp_checkpoint_

% disp('rescaling trf vars...')

%% TRF ANALYSIS
%TODO: replace all instances of load_checkpoint with appropriate 
% registry-validated loading
%function
%% check if variables for current config can be preloaded

% function starts here...
preload_stats_null=false;
preload_stats_obs=false;
preload_model=false;
%% preload trf results (TODO: DEBUG EVERYTHING BELOW THIS LINE)

trf_checkpoint_=load_checkpoint(trf_config);
if ~isempty(trf_checkpoint_)
    if isfield(trf_checkpoint_,'stats_obs')
        stats_obs=trf_checkpoint_.stats_obs;
        preload_stats_obs=true;
    end
    if isfield(trf_checkpoint_,'stats_null')
        stats_null=trf_checkpoint_.stats_null;
        preload_stats_null=true;
    end
    if isfield(trf_checkpoint_,'model')
        model=trf_checkpoint_.model;
        preload_model=true;
    end
else
    disp('no data for trf checkpoint with trf_config:')
    disp(trf_config)
    disp('running analysis')
end
clear trf_checkpoint_
% note: we don't necessarily need these if results can preload from
% checkpoint but it doesn't take very long AND we definitely will def need
% to set up any trf training so just set it up in case needed
disp('rescaling trf vars.')
% will defnitely need rescaled vars
[stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
    trf_config);
% if none([preload_stats_null,preload_stats_null,preload_model])
%     disp('rescaling trf vars.')
%     % will defnitely need rescaled vars
%     [stim,preprocessed_eeg]=rescale_trf_vars(stim,preprocessed_eeg, ...
%         trf_config,preprocess_config);
% else
% end
%%
if ~preload_stats_obs && trf_config.crossvalidate
    stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config);
    % fprintf('saving stats_obs to %s...\n',trf_config.paths.output_dir)
    save_checkpoint(stats_obs,trf_config);
end
%%

% if exist(trf_config.paths.model_metric_path,'file')
%     % load new-format
%     fprintf('checking %s for checkpoint data.\n',trf_config.model_metric_path)
%     % will load both stats_obs and stats_null (if they exist... at least
%     % obs should if file does)
%     metric_checkpoint=load_checkpoint(trf_config.model_metric_path,trf_config);
%     % if isempty(metric_checkpoint)
%     % if ismember('stats_obs',who('-file',trf_config.model_metric_path))
%     if ismember('stats_obs',fieldnames(metric_checkpoint))
%         stats_obs=metric_checkpoint.stats_obs;
%         preload_stats_obs=true;
%         % if ismember('stats_null',who('-file',trf_config.model_metric_path))
%         if ismember('stats_null',fieldnames(metric_checkpoint))
%             stats_null=metric_checkpoint.stats_null;
%             preload_stats_null=true;
%         end
%     end
%     clear metric_checkpoint
% end
%% THIS NEEDS TO BE UPDATED SO THAT BEST_LAM IS ADDED TO MODEL INSTEAD OF CONFIG 
if trf_config.do_lambda_optimization
    % find best lambda from crossvalidation
    %NOTE: assuming that all conditions are together when optimizing
    %lambda and separate when not (but lambda is fixed)
    [best_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs, ...
        trf_config);
    trf_config.best_lam=best_lam;
else
    disp(['need to figure out what to do here ... ' ...
        'if we want a specific value that isnt the same as ' ...
        'optimization result?'])
    trf_config.best_lam=fetch_optimized_lam(trf_config);
end
%%
if ~preload_model
    model=train_model(stim,preprocessed_eeg,trf_config);
    fprintf('saving model to %s\n...',trf_config.trf_model_path)
    % save(trf_config.trf_model_path,'model','trf_config');
    % NOTE: after this save checkpoint, best_lam should be included in
    % trf_config BUT probably instead what will happen is our checkpoint
    % validation will identify this as a non-identical config and add a new
    % checkpoint?
    % RE: above - seems like it falls under case 1 in save_checkpoint,
    % which does not alter in-file config... which might be ok for now
    save_checkpoint(model,trf_config);
end
%%
if do_nulltest && ~preload_stats_null
    stats_null=get_nulldist(stim,preprocessed_eeg,trf_config);
    % error('stuff below should take place in save_checkpoint...')
    % fprintf('append-saving stats_null to %s...\n',trf_config.model_metric_path)
    % save(trf_config.model_metric_path,'stats_null','-append')
    save_checkpoint(stats_null,trf_config);
    % disp(['TODO: not saving stats_null to bypass save_checkpoint error.. ' ...
    %     'determine if thats a problem'])
else
    disp('not doing null test (or preloaded previous results)')
end

%%
if do_nulltest
    nulltest_plot_wrapper(stats_obs,stats_null,trf_config,preprocessed_eeg)
end



end
%% Helpers
function best_lam=fetch_optimized_lam(trf_config)
warning('this hasnt been updated')
%TODO: use registry instead
% best_lam=fetch_optimized_lam(trf_config)
% pull best_lam from trf_config associated with condition-agnostic trf
    load(trf_config.model_metric_path,'stats_obs');
    [best_lam,~,~]=get_best_lam(stats_obs(1,1),trf_config);

end
function nulltest_plot_wrapper(stats_obs,stats_null,trf_config,preprocessed_eeg)
% nulltest_plot_wrapper(stats_obs,stats_null,trf_config,preprocessed_eeg)
%NOTE: we got rid of r_null/r_obs nuisance vars which should be computed in
%plotting function anyway
fav_chn_idx=85;
subj=trf_config.subj;
conditions=unique(preprocessed_eeg.cond)';
% best_lam=trf_config.best_lam;
% bl_indx=find(trf_config.lam_range==best_lam);
%TODO: add check here to ensure thing computed outside of this function 
% in "get_best_lam" agrees and we're not plotting some other random shit
% running this in separate conditions case is not working out too well...
% best chn/best lam should 
% [best_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs,trf_config);
% if best_lam~=trf_config.best_lam
%     error('best lambda value does not agree with trf_config')
% end

%TODO: double-check averaging across trials correctly in separate
%conditions case (especially once it contains multiple versions in last dimension)
disp('cc indexing below might need double checking...')
%NOTE: although it's possible I'm wrong and will work fine so long as we
%select correct sub-structures out during load_checkpoint... i think
%cc_trials_idx below comes from first array dimension referenccing
%individual trials, not our configs
if trf_config.separate_conditions
    for cc=conditions
        [best_lam,~,best_chn_idx]=get_best_lam(stats_obs(1,cc),trf_config);
        % cc_trials_idx=find(preprocessed_eeg.cond==cc);
        %TODO: check size resulting from squeeze agrees with plotting
        %commands
        % disp('r_null calculation below might be incorrect...?')
        r_null=squeeze(mean([stats_null(1,cc,:).r],1));
        % fprintf('_rnull size correct?\n')
        % disp(size(r_null))
        r_obs=squeeze(mean(stats_obs(1,cc,:).r));

        tit_str_temp=sprintf(['subj %d, cond %d, fav chn (%d) permutation ' ...
            'test - \\lambda %.3g'],subj,cc,fav_chn_idx,best_lam);
        nulltest_fig_helper(r_null,r_obs,fav_chn_idx,tit_str_temp)
        clear tit_str_temp

        tit_str_temp=sprintf(['subj %d, cond %d, best chn (%d) permutation ' ...
            'test - \\lambda %.3g'],subj,cc,best_chn_idx,best_lam);
        nulltest_fig_helper(r_null,r_obs,best_chn_idx,tit_str_temp)
        clear tit_str_temp
        
    end

else
    [best_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs,trf_config);
    % NOTE: temporary fix to just convert comma-sep list into array for
    % getting r_null... will need to adapt once we fix this
    r_null=squeeze(mean([stats_null.r],1));
    r_obs=squeeze(mean(stats_obs.r(:,best_lam_idx,:)));
    tit_str_temp=sprintf(['subj %d, all-cond , best chn (%d) permutation ' ...
        'test - \\lambda %.3g'],subj,best_chn_idx,best_lam);
    nulltest_fig_helper(r_null,r_obs,best_chn_idx,tit_str_temp)
    clear tit_str_temp

    tit_str_temp=sprintf(['subj %d, all-cond , fav chn (%d) permutation ' ...
        'test - \\lambda %.3g'],subj,fav_chn_idx,best_lam);
    nulltest_fig_helper(r_null,r_obs,fav_chn_idx,tit_str_temp)
    clear tit_str_temp
    % figure
    % hist(r_null(:,fav_ch_idx))
    % title(sprintf('subj %d, chn %d null distribution - \\lambda %.3g',subj,fav_ch_idx,best_lam))
    % figure
    % ecdf(r_null(:,fav_ch_idx))
    % hold on
    % plot(repmat(r_obs(fav_ch_idx),1,2),ylim,'r')
    % title(sprintf('subj %d, chn %d permutation test - \\lambda %.3g',subj,fav_ch_idx,best_lam))
end
end

function nulltest_fig_helper(r_null,r_obs,plot_chn,tit_str)
% nulltest_fig_helper(r_null,r_obs,plot_chn,tit_str)
    ylim=[0 1];
    figure
    hist(r_null(:,plot_chn))
    title(tit_str)
    figure
    ecdf(r_null(:,plot_chn))
    hold on
    plot(repmat(r_obs(plot_chn),1,2),ylim,'r')
    title(tit_str)
end

function stats_null=get_nulldist(stim,preprocessed_eeg,trf_config)
    disp('running permutation test...')
    msg = 0;
    n_permutations=1000;
    fs=preprocessed_eeg.fs;
    cv_tmin_ms=trf_config.cv_tmin_ms;
    cv_tmax_ms=trf_config.cv_tmax_ms;
    conditions=unique(preprocessed_eeg.cond)';
    model_lam=trf_config.best_lam;

    for n_perm = 1:n_permutations
        if ~mod(n_perm,50)
            fprintf(repmat('\b',msg,1))
            msg = fprintf('iteration #%0.4d\n',n_perm);
        end
        resp_shuf = preprocessed_eeg.resp;
        % shuffle stim/responses within conditions
        for cc = 1:3
            I = find(preprocessed_eeg.cond==cc);
            I2 = I(randperm(length(I)));
            resp_shuf(I2) = preprocessed_eeg.resp(I);
        end

        if trf_config.separate_conditions
            % stats_null=cell(3,1);
            % r_null=cell(3,1);
            for cc=conditions
                % fprintf('null TRF for condition %d...\n',cc)
                cc_mask=preprocessed_eeg.cond==cc;
                stats_null(1,cc,n_perm)=mTRFcrossval(stim(cc_mask), ...
                    resp_shuf(cc_mask),fs,1,cv_tmin_ms,cv_tmax_ms, ...
                    model_lam,'Verbose',0);
            end
        else
            stats_null(1,1,n_perm) = mTRFcrossval(stim,resp_shuf,fs,1,cv_tmin_ms, ...
                cv_tmax_ms,model_lam,'Verbose',0);
        end
    end
end

function model=train_model(stim,preprocessed_eeg,trf_config)
% model=train_model(stim,preprocessed_eeg,model_lam,trf_config)
model_lam=trf_config.best_lam;
fprintf('training model using lambda=%0.2g\n',model_lam)

tmin_ms=trf_config.tmin_ms;
tmax_ms=trf_config.tmax_ms;
resp=preprocessed_eeg.resp;
fs=preprocessed_eeg.fs;
if trf_config.separate_conditions
        disp('evaluating trf models separately per condition')
        conditions=unique(preprocessed_eeg.cond)';
        for cc=conditions
            cc_mask=preprocessed_eeg.cond==cc;
            fprintf('TRF for condition %d...\n',cc)
            model(1,cc)=mTRFtrain(stim(cc_mask),resp(cc_mask),fs,1, ...
                tmin_ms,tmax_ms,model_lam,'Verbose',1);
        end
    else
        model = mTRFtrain(stim,resp,fs,1,tmin_ms,tmax_ms,model_lam,'Verbose',1);
end
end

function [model_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs,trf_config)
% [model_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs)
% NOTE this function expects stats_obs to have [1,1] size and doesnt work
% otherwise.. probably should clean this up in load_checkpoint but
% addressing here for now by assuming non-(1,1) values in stats_obs are empty
    if ~isequal(size(stats_obs),[1,1])
        stats_obs=stats_obs(1,1);
    end
    r_avg_trials=squeeze(mean(stats_obs.r,1));
    % get max across electrodes for each lambda
    r_max_electrodes=squeeze(max(r_avg_trials,[],2));
    % get indices of max r-value 
    if isfield(trf_config,'best_lam')&&size(stats_obs.r,2)==1
        % no idea why we did this
        disp('TODO: fix this workaround...')
        best_lam_idx=nan;
        model_lam=trf_config.best_lam;
        [~,best_chn_idx]=max(r_avg_trials);
    else
        [~,best_lam_idx]=max(r_max_electrodes);
        model_lam=trf_config.lam_range(best_lam_idx);
        [~,best_chn_idx]=max(r_avg_trials(best_lam_idx,:));
    end
    
    
    
end

function restart_bool=check_restart(event_trials)
% helper function to check if there are excess trials from restarting
% should work even if first trial is not trial 1
if any(diff(event_trials)<1)
    restart_bool=true;
else
    restart_bool=false;
end
end

function stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config)
% function stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config,preprocess_config)
fprintf('TODO: double-check structure indexing in this function (crossval_wrapper)...\n')
% stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config)
resp=preprocessed_eeg.resp;
if trf_config.do_lambda_optimization
    cv_lam=trf_config.lam_range;
else
    % warning('this hasnt been updated to load ')
    trf_config.best_lam=fetch_optimized_lam(trf_config);
    cv_lam=trf_config.best_lam;
end
fs=preprocessed_eeg.fs;
tmin_ms=trf_config.tmin_ms;
tmax_ms=trf_config.tmax_ms;

fprintf('running crossval with lam val(s):\n')
fprintf('%0.2g \n',cv_lam)

if trf_config.separate_conditions
        % gives 1,2,3 for either slowest to fastest 
        % 1.5,1,.67 time compress factor or smallest to largest time
        
        conditions=unique(preprocessed_eeg.cond)';
        for cc=conditions
            cc_mask=preprocessed_eeg.cond==cc;
            stats_obs(1,cc)=mTRFcrossval(stim(cc_mask),resp(cc_mask),fs,1, ...
                tmin_ms,tmax_ms,cv_lam,'Verbose',0);
        end
else
    stats_obs = mTRFcrossval(stim,resp,fs,1,tmin_ms,tmax_ms,cv_lam, ...
        'Verbose',0);
end
end


function [EEG, cond,preprocessed_eeg]=clean_false_starts(EEG,cond,preprocessed_eeg)
% [EEG, cond,preprocessed_eeg]=clean_false_starts(EEG,cond,preprocessed_eeg)
    %TODO: check if multiple false starts can be accounted for
    % if multiple restarts, grab the final one
    start_iis=[0 diff(preprocessed_eeg.trials)];
    last_start_ii=find(start_iis<1,1,'last');
    max_repeated_trial_type=preprocessed_eeg.trials(last_start_ii-1);
    preprocessed_eeg.trials=preprocessed_eeg.trials(last_start_ii:end);
    % remove false start trials from EEG structure so epoch won't
    % return multiple trials for those
    false_start_end_ii=find([EEG.event.type]==max_repeated_trial_type,1,'first');
    % stupid loop because idk how to assign multiple struct array
    % vals otherwise
    for rm_trial=1:false_start_end_ii
        EEG.event(rm_trial).type=3000;
    end
    %TODO: verify what line below does
    %NOTE: it doesn't seem to do anything... but I'm thinking it might be
    %necessary to keep cond limited to the relevant trials when cleaning
    %false starts... so i put it there for now
    cond=cond(preprocessed_eeg.trials,1);
    
end

function preprocessed_eeg=preprocess_eeg(preprocess_config)
% TODO: check if cond variable output turns out the way we were
% expecting...
% preprocessed_eeg=preprocess_eeg(preprocess_config)
    %  load bdf data and experimental conditions info
    if preprocess_config.fs/2<=preprocess_config.bpfilter(2)
        error(['bpfilter lowpass cutoff (Hz %d) is above nyquist for ' ...
            'output fs (%d Hz)'],preprocess_config.bpfilter(2),round(preprocess_config.fs/2))
    end
    EEG = pop_biosig(preprocess_config.paths.bdffile,preprocess_config.opts{:});
    load(preprocess_config.paths.behfile,'m')
    cond = round(m(:,1),2);
    cond(cond>1) = 3;
    cond(cond==1) = 2;
    cond(cond<1) = 1;

    % remove mastoids
    EEG = pop_select(EEG,'nochannel',preprocess_config.nchan+(1:2));
    warning('no chanlocs file added until verifying which is correct and how to enter it.')
    % EEG = pop_chanedit(EEG,'load',{preprocess_config.paths.chanlocs_path,'filetype','xyz'});
    EEG.urchanlocs = EEG.chanlocs;
    
    if preprocess_config.interpBadChans
        warning(['since resample was moved to end of pipeline finding bad' ...
            ' chans will be really slow...'])
        % note that resp will have the interpolated channels but
        % original bdf will not
        disp('finding bad channels')
        %TODO: decide if we want to store badChans in preprocess_config
        %or with preprocessed mat file
        preprocessed_eeg.badChans=findBadChans(EEG.data'); 
        if ~isempty(preprocessed_eeg.badChans)
            EEG=eeg_interp(EEG,preprocessed_eeg.badChans);
        end
    end
    % filter to frequency band of interest
    disp('filtering with bpfilter:')
    disp(preprocess_config.bpfilter)
    hd_hp = getHPFilt(EEG.srate,preprocess_config.bpfilter(1));
    EEG.data = filtfilthd(hd_hp,EEG.data')';
    hd_lp = getLPFilt(EEG.srate,preprocess_config.bpfilter(2));
    EEG.data = filtfilthd(hd_lp,EEG.data')';
    % note: assumes new fs is integer-multiple of starting fs and that
    % bpfilter sufficiently doubles as antialias filter
    EEG=pop_downsample_noAA(EEG,preprocess_config.fs);




    % Epoching
    preprocessed_eeg.trials = [EEG.event.type];
    % why did aaron choose 100 in particular rather than preprocess_config.n_trials to
    % begin with?
    preprocessed_eeg.trials(preprocessed_eeg.trials>preprocess_config.n_trials+1) = [];
    bdf_triggers_missing=any(diff(preprocessed_eeg.trials)>1);
    % filter out missing trials from cond
    if bdf_triggers_missing
        % cond might still have recorded, which will mess indexing
        % downstream
        % note that this block ASSUMES cond has the missing trial, which
        % may not necessarily always be the case...
        expected_trials=1:preprocess_config.n_trials;
        % note we probably should save this in preprocess_config... will be
        % a hassle right now so ignoring that
        missing_trials=setdiff(preprocessed_eeg.trials,expected_trials);
        valid_trials=ismember(expected_trials,preprocessed_eeg.trials);
        cond=cond(valid_trials);
    end
    % remove repeated trials from EEG structure first, if any
    has_false_start=check_restart(preprocessed_eeg.trials);
    if has_false_start
        [EEG,cond,preprocessed_eeg]=clean_false_starts(EEG,cond,preprocessed_eeg);
    end

    
    % check if last condition is slow, otherwise need to pad EEG.data
    % so pop_epoch works (last trial needs to be within epoch_dur
    % boundary)
    if cond(end)~=3
        % slow condition is fine, others need padding
       
        ns_pad=floor((preprocess_config.epoch_dur)*EEG.srate);
        EEG.data=[EEG.data, zeros(preprocess_config.nchan,ns_pad)];
    end

    if ~isempty(preprocess_config.stim_delay_time)
        EEG=add_speech_delay(EEG,preprocess_config);
    end
    
    EEG = pop_epoch(EEG, ...
        mat2cell(preprocessed_eeg.trials,1, ...
        ones(1,numel(preprocessed_eeg.trials))),[0 preprocess_config.epoch_dur]);

    resp = cell(1,size(EEG.data,3));
    for tt = 1:size(EEG.data,3)
        resp{1,tt} = EEG.data(:,:,tt)';
    end
    
    preprocessed_eeg.resp=resp;
    preprocessed_eeg.cond=cond;
    preprocessed_eeg.hd_hp=hd_hp;
    preprocessed_eeg.hd_lp=hd_lp;
    preprocessed_eeg.fs=EEG.srate;
end

function preprocessed_eeg=remove_excess_epoch(stim,preprocessed_eeg)
% trim resp from epoch_dur to stim_dur
fprintf('removing excess samples from pop_epoch\n')
for tt = 1:numel(stim)
    preprocessed_eeg.resp{tt} = preprocessed_eeg.resp{tt}(1:size(stim{tt},1),:);
end

end


function EEG=add_speech_delay(EEG,preprocess_config)
% delayed_EEG=add_speech_delay(EEG,preprocess_config)
    % shift trial starts by 1 second (speech_delay value in expriment script)      
    
    
    n_delay_samples=preprocess_config.stim_delay_time*EEG.srate; 
    fprintf('adding %0.2fs /%d samples\n',preprocess_config.stim_delay_time,n_delay_samples)

    % TODO: ASK AARON HOW TO DO THIS WITHOUT LOOPS FOR THE LOVE OF
    % ALL THAT IS HOLY
    % RE: above - I did and his idea didn't work but there must be
    % another way....
    for trial_indx=find([EEG.event.type]<=preprocess_config.n_trials)
        EEG.event(trial_indx).latency=EEG.event(trial_indx).latency+n_delay_samples;
    end
end