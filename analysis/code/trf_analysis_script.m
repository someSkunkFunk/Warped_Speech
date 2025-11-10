clearvars -except user_profile boxdir_mine boxdir_lab
clc
%NOTES:
% n_trials is sortof redundant now that m is a part of preprocess_config...
% TODO: could benefit from dedicated function to check EEGlab structure
% events make sense... or could make it a point to visually inspect each
% subject
% for subj=[2:7,9:22]
% for subj=[9, 12, 96, 97, 98] %note: do this after 12 and up run
% successfully for all conditions
% for subj=[9,12,96,98]
% for subj=[7,9:21]
% for subj=[2:7,9:22,96,98] % next need to run for separate conditions on all subjects
% for subj=[2:7,9:23,96,98]
for subj=[23]
clearvars -except user_profile boxdir_mine boxdir_lab subj
close all
%% setup analysis
trf_analysis_params;
if update_configs
    disp('**********UPDATING CONFIGS - NOT RUNNING ANALYSIS SCRIPT***********')
    % assumes these exist already and just need to add fields not
    % previously saved
    if ~trf_config.separate_conditions
        % should already be updated when updating sep conditions configs
        disp('updating preprocess config')
        update_rehash_config(preprocess_config)
        disp('done.')
    end
    disp('updating trf config')
    update_rehash_config(trf_config)
    disp('done.')  
    continue
end
%% check if data exists already...
if overwrite 
    pp_checkpoint_=[];
    trf_checkpoint_=[];
else
    pp_checkpoint_=load_checkpoint(preprocess_config);
    trf_checkpoint_=load_checkpoint(trf_config);
end


%% preprocess from raw (bdf)
if isempty(pp_checkpoint_)
    fprintf('processing from bdf...\n')
    preprocessed_eeg=preprocess_eeg(preprocess_config);
    stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
    % trim resp to have same durations as stim (only need to do during
    % preprocessing)
    preprocessed_eeg=remove_excess_epoch(stim,preprocessed_eeg);
    fprintf('saving to %s\n',preprocess_config.paths.output_dir)
    save_checkpoint(preprocessed_eeg,preprocess_config,overwrite)
    preload_preprocessed=false;
else
    disp('found existing preprocessed data - loading')
    preprocessed_eeg=pp_checkpoint_.preprocessed_eeg;
    stim=load_stim_cell(trf_config.paths.envelopesFile,preprocessed_eeg.cond,preprocessed_eeg.trials);
    preload_preprocessed=true;
end
clear pp_checkpoint_

% disp('rescaling trf vars...')

%% TRF ANALYSIS

%% check if variables for current config can be preloaded

% function starts here...
preload_stats_null=false;
preload_stats_obs=false;
preload_model=false;
%% preload trf results (TODO: DEBUG EVERYTHING BELOW THIS LINE)

if ~isempty(trf_checkpoint_)
    disp('loading trf checkpoint data...')
    if isfield(trf_checkpoint_,'stats_obs')
        stats_obs=trf_checkpoint_.stats_obs;
        preload_stats_obs=true;
        disp('stats_obs preloaded.')
    end
    if isfield(trf_checkpoint_,'model')
        model=trf_checkpoint_.model;
        preload_model=true;
        disp('model preloaded.')
    end
    if isfield(trf_checkpoint_,'stats_null')        
        stats_null=trf_checkpoint_.stats_null;
        preload_stats_null=true;
        disp('stats_null preloaded.')
    end
else
    disp('no data preloaded for trf checkpoint with trf_config:')
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

%%
if (~preload_stats_obs && trf_config.crossvalidate)
    stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config,train_params);
    save_checkpoint(stats_obs,trf_config,overwrite);
    % fprintf('saving stats_obs to %s...\n',trf_config.paths.output_dir)
   %%%%%NOTE: DO NOT SAVE RESULT YET BECAUSE CONFIG ISNT UPDATED WITH
   %%%%%BEST_LAM, WHICH WILL RESULT IN A DIFFERENT HASH VALUE IN
   %%%%%SAVE_CHECKPOINT
end

%%
if ~trf_config.separate_conditions
    % otherwise it gets set in trf_analysis_params
    % note if we want to crossvalidate with manual lam value this will
    % cause bug
    train_params.best_lam=plot_lambda_tuning_curve(stats_obs,trf_config);
else
    plot_lambda_tuning_curve(stats_obs,trf_config);
end
if ~preload_model
    model=train_model(stim,preprocessed_eeg,trf_config,train_params);
    save_checkpoint(model,trf_config,overwrite);
end
%%
if do_nulltest && ~preload_stats_null
    %note this is dumb and clunky but avoids error when model is preloaded
    if ~trf_config.separate_conditions&&~isfield(train_params,'best_lam')
        % otherwise it gets set in trf_analysis_params
        train_params.best_lam=plot_lambda_tuning_curve(stats_obs,trf_config,85);
    end
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
    nulltest_plot_wrapper(stats_obs,stats_null,trf_config)
end
end
%% Helpers
function nulltest_plot_wrapper(stats_obs,stats_null,trf_config)
% nulltest_plot_wrapper(stats_obs,stats_null,trf_config)

fav_chn_idx=85;
subj=trf_config.subj;
% if trf_config.separate_conditions
%     % conditions=unique(preprocessed_eeg.cond)';
%     % conditions={' - fast - ',' - original - ',' - slow - '};
%     conditions=cellfun(@(x) sprintf('- %s -',x),trf_config.conditions);
% else
%     conditions={''};
% end
conditions=cellfun(@(x) sprintf('- %s -',x),trf_config.conditions,'UniformOutput',false);
%TODO: double-check averaging across trials correctly in separate
%conditions case (especially once it contains multiple versions in last dimension)
disp('cc indexing below might need double checking...')
%NOTE: although it's possible I'm wrong and will work fine so long as we
%select correct sub-structures out during load_checkpoint... i think
%cc_trials_idx below comes from first array dimension referenccing
%individual trials, not our configs
best_lam=train_params.best_lam;
for cc=1:length(conditions)
    if trf_config.separate_conditions
    % for cc=conditions
        % [best_lam,~,best_chn_idx]=get_best_lam(stats_obs(1,cc),trf_config);
    r_null=squeeze(mean([stats_null(1,cc,:).r],1));
    r_obs=squeeze(mean(stats_obs(1,cc,:).r,1));

    else
    % [best_lam,best_lam_idx,best_chn_idx]=get_best_lam(stats_obs,trf_config);
        best_lam_idx=find(trf_config.lam_range==best_lam);
        r_null=squeeze(mean([stats_null.r],1));
        r_obs=squeeze(mean(stats_obs.r(:,best_lam_idx,:),1));
   
    end
    % note: at this point, r_obs be chns-by-1 vector in either case
    [~,best_chn_idx]=max(r_obs);
tit_str_temp=sprintf(['subj %d %s best chn (%d) permutation ' ...
    'test - \\lambda %.3g'],subj,conditions{cc},best_chn_idx,best_lam);
nulltest_fig_helper(r_null,r_obs,best_chn_idx,tit_str_temp)
clear tit_str_temp

tit_str_temp=sprintf(['subj %d %s fav chn (%d) permutation ' ...
    'test - \\lambda %.3g'],subj,conditions{cc},fav_chn_idx,best_lam);
nulltest_fig_helper(r_null,r_obs,fav_chn_idx,tit_str_temp)
clear tit_str_temp
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

function stats_null=get_nulldist(stim,preprocessed_eeg,trf_config,train_params)
% stats_null=get_nulldist(stim,preprocessed_eeg,trf_config)
%TODO: prettify params
    disp('running permutation test...')
    msg = 0;
    n_permutations=1000;
    fs=preprocessed_eeg.fs;
    tmin_ms=trf_config.tmin_ms;
    tmax_ms=trf_config.tmax_ms;
    conditions=unique(preprocessed_eeg.cond)';
    best_lam=train_params.best_lam;

    for n_perm = 1:n_permutations
        if ~mod(n_perm,50)
            fprintf(repmat('\b',msg,1))
            msg = fprintf('iteration #%0.4d\n',n_perm);
        end
        resp_shuf = cell(size(preprocessed_eeg.resp));
        % shuffle stim/responses within conditions
        
        switch trf_config.experiment
            case 'fast-slow'
                for cc=conditions
                    % permute condition labels
                    I = find(preprocessed_eeg.cond==cc);
                    I2 = I(randperm(length(I)));
                    resp_shuf(I2) = preprocessed_eeg.resp(I);
                end
            case 'reg-irreg'
                % reg/irreg stimuli don't have consistent durations even within
                % condition
                % -> circularly shift responses
                shift_amt=randi(min(cellfun('length',preprocessed_eeg.resp)));
                resp_shuf=cellfun(@(x) circshift(x,shift_amt,1),preprocessed_eeg.resp,'UniformOutput',false);

        end

        
        if trf_config.separate_conditions
            % stats_null=cell(3,1);
            % r_null=cell(3,1);
            for cc=conditions
                % fprintf('null TRF for condition %d...\n',cc)
                cc_mask=preprocessed_eeg.cond==cc;
                stats_null(1,cc,n_perm)=mTRFcrossval(stim(cc_mask), ...
                    resp_shuf(cc_mask),fs,1,tmin_ms,tmax_ms, ...
                    best_lam,'Verbose',0);
            end
        else
            stats_null(1,1,n_perm) = mTRFcrossval(stim,resp_shuf,fs,1,tmin_ms, ...
                tmax_ms,best_lam,'Verbose',0);
        end
    end
end

function model=train_model(stim,preprocessed_eeg,trf_config,train_params)
% model=train_model(stim,preprocessed_eeg,model_lam,trf_config,train_params)
disp('training model with params:')
disp(train_params)

tmin_ms=train_params.tmin_ms;
tmax_ms=train_params.tmax_ms;
best_lam=train_params.best_lam;
resp=preprocessed_eeg.resp;
fs=preprocessed_eeg.fs;
if trf_config.separate_conditions
        disp('evaluating trf models separately per condition')
        conditions=unique(preprocessed_eeg.cond)';
        for cc=conditions
            cc_mask=preprocessed_eeg.cond==cc;
            fprintf('TRF for condition %d...\n',cc)
            model(1,cc)=mTRFtrain(stim(cc_mask),resp(cc_mask),fs,1, ...
                tmin_ms,tmax_ms,best_lam,'Verbose',1);
        end
    else
        model = mTRFtrain(stim,resp,fs,1,tmin_ms,tmax_ms,best_lam,'Verbose',1);
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

function stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config,train_params)
% stats_obs=crossval_wrapper(stim,preprocessed_eeg,trf_config,preprocess_config)

resp=preprocessed_eeg.resp;
if trf_config.do_lambda_optimization
    cv_lam=trf_config.lam_range;
else
    warning('is this really what you want crossval_wrapper to do?')
    % NOTE: feeling like lambda optimization is a pointless parametr and
    % more cumbersome than needed
    cv_lam=train_params.best_lam;
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
        % is this actually valid??
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
    m=preprocess_config.m;
    % m_=unique(m(:,1:2)','rows','sorted');
    % m_=m(:,1:2)';
    m_=[all(m(:,1)==1),all(m(:,2)==0)];
    %note: we could just use experiment var here but this also functions as
    %check that conditions are what's expected
    switch preprocess_config.experiment
        case 'reg-irreg'
            % double-check conditions
            if isequal(m_,[1,0])
                cond=round(m(:,2))+2;
            else
                error('conditions in m dont match reg-irreg pattern.')
            end
        case 'fast-slow'
            % double-check conditions
            if isequal(m_,[0,1])
                cond = round(m(:,1),2);
                cond(cond>1) = 3;
                cond(cond==1) = 2;
                cond(cond<1) = 1;
            else
                error('conditions in m dont match fast-slow pattern')
            end
        otherwise
            error('(preprocess_eeg) set of conditions unexpected.')
    end
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
    warning('see comment below this line...')
    % NOTE: no longer relying on psychport trial number trigger and instead
    % using click trigger since it is more accurate (and sometimes trial 
    % number trigger overlaps with click trigger) - should validate that
    % this is retroactively compatible with subjs 2-22
    function EEG=clean_eeg_events(EEG,ntrials)
        fix_overlap=false;
        all_triggs=[EEG.event(:).type];
        click_trigger=2048;
        % keep only triggers corresponding with sound clicks that
        % don't overlap with other triggers
        click_trigg_idx=find(all_triggs==click_trigger);
        overlap_trigg_idx=find((all_triggs>click_trigger)& ...
            (all_triggs<=ntrials+click_trigger));
        psych_trigg_idx=find(all_triggs<=ntrials);
        if any(overlap_trigg_idx)
            warning(['%d triggers overlap with click trigger, ' ...
                'will attempt to disentangle them but double check ' ...
                'result.'],numel(overlap_trigg_idx))
            fix_overlap=true;
            overlap_latencies=[EEG.event(overlap_trigg_idx).latency];
            rm_overlap_mask=false(size(overlap_latencies));
            click_latencies=[EEG.event(click_trigg_idx).latency];
            rm_click_mask=false(size(click_latencies));
            psych_latencies=[EEG.event(psych_trigg_idx).latency];
            rm_psych_mask=false(size(psych_latencies));
        end
        switch preprocess_config.use_triggers
            case 'click'
                disp('removing events unrelated to click triggers...')
                if isempty(click_trigg_idx)&&isempty(overlap_trigg_idx)
                    error('no click triggers found - check events')
                elseif sum([numel(click_trigg_idx),numel(overlap_trigg_idx)])<ntrials
                    
                    error(['there are less click triggers than specified \n' ...
                        'number of trials:\n' ...
                        '%d clicks, vs %d trials expected'], ...
                        sum(numel(click_trigger),numel(overlap_trigg_idx)),ntrials)
                end
                
                if length(click_trigg_idx)==ntrials&&isempty(overlap_trigg_idx)        
                    % best-case, just keep those
                    EEG.event=[EEG.event(click_trigg_idx)];
                    types=1:ntrials;
                    
                elseif fix_overlap
                    % note: muted case expression above because some
                    % triggers may not overlap at all in which case n
                    % overlap will be less than ntrials but we still want
                    % to handle overlaps the same - keep the click it
                    % overlaps with if it occurs before the overlapped
                    % trigger or keep the overlapped trigger if click
                    % registers after it
                    for ii=1:length(click_latencies)
                        % get delay relative to overlap trigger
                        click_delays=click_latencies(ii)-overlap_latencies;
                        % overlapping triggers should be 1 sample adjacent
                        % from each other...
                        late_clicks=click_delays<=1&click_delays>0;
                        early_clicks=click_delays>=-1&click_delays<0;
                        % clicks that are a sample late are no good
                        if any(late_clicks) 
                            % ^ note that we're assuming at most one such
                            % case is possible
                            rm_click_mask(ii)=true;
                        % clicks that are a sample early mean overlap
                        % trigger is no good
                        elseif any(early_clicks)
                        % ^ note that we're assuming at most one such
                        % case is possible
                            rm_overlap_mask(early_clicks)=true;
                        end
                    end
                    overlap_trigg_idx(rm_overlap_mask)=[];
                    click_trigg_idx(rm_click_mask)=[];
                    keep_triggs=sort([overlap_trigg_idx,click_trigg_idx]);
                    EEG.event=[EEG.event(keep_triggs)];
                    types=[EEG.event.type]-click_trigger;
                    
                    % check we have correct number of trials now
                    if (numel(overlap_trigg_idx)+numel(click_trigg_idx))~=ntrials
                        % clean_false_starts might be able to fix the
                        % problem if it arises but if not the issue could
                        % be something else...
                        error(['after  attempting to disentangle \n' ...
                            'overlapping triggers, number of trials\n' ...
                            'is incorrect, could be due,\n' ...
                            'to restart. Since we need event index to ' ...
                            'match trial number to correct type, this cant' ...
                            'be fixed by clean_false_starts later... \n'])
                    else
                        if any(types==0)
                            % by now remaining non-overlapping click 
                            % triggers index should match their trial number
                            types(types==0)=find(types==0);
                        end
                    
                    end
                end
                % replace trigger values with trial numbers (assuming no missing
                % trials)
                types=num2cell(types);
                [EEG.event(:).type]=types{:};
            case 'psychportaudio'
                disp('removing events unrelated to psychportaudio triggers...')
                % assumes trial num triggers are all present and not
                % overlapping with click triggers (or any other triggers)
                % TODO: cover case for subj 6 where trial trigger is missing...
                % for subj 6 - there are no click triggers we can use but the
                % psychaudio trigger definitely overlaps with a 2048 we could use!

                %thus should follow process:
                %1. remove triggers below ntrials and above 2048
                % if any above 2048 and below ntrials less than ntrials,
                % replace large one with missing trial numbers

                %also needs to handle case where experiment restarts
                % if fix_overlap
                %     % this just removes events with trigger value greater
                %     % than 
                %     rm_overlap_mask=~ismember(all_triggs(overlap_trigg_idx)-click_trigger,1:ntrials);
                % end
                types_renum=num2cell([EEG.event(overlap_trigg_idx).type]-click_trigger);
                [EEG.event(overlap_trigg_idx).type]=types_renum{:};
                keep_triggs=sort([overlap_trigg_idx,psych_trigg_idx]);
                EEG.event=[EEG.event(keep_triggs)];
                % seems like sometimes there's overlapping triggers that
                % follow... need to remove them while
                double_triggs=find(diff([EEG.event.type])==0)+1;
                EEG.event(double_triggs)=[];
            otherwise
                warning('invalid triggers chosen: %s',preprocess_config.use_triggers)
        end

    end
    % verify number of events matches number of trials
    
    EEG=clean_eeg_events(EEG,preprocess_config.n_trials);
   
    preprocessed_eeg.trials = [EEG.event.type];
    % why did aaron choose 100 in particular rather than preprocess_config.n_trials to
    % begin with?
    % note: line below is now unnecessary since clean_eeg_events should
    % only leave trial triggers... but we should verify that only one such
    % even per trial remains for all subjects
    warning('see comment above...')
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
        disp('missing trials detected:')
        disp(missing_trials)
    end
    % remove repeated trials from EEG structure first, if any
    has_false_start=check_restart(preprocessed_eeg.trials);
    if has_false_start
        warning('false starts detected... attempting to fix')
        [EEG,cond,preprocessed_eeg]=clean_false_starts(EEG,cond,preprocessed_eeg);
    end

    
    function EEG=add_speech_delay(EEG,preprocess_config)
    % delayed_EEG=add_speech_delay(EEG,preprocess_config)
        % shift trial starts by 1 second (speech_delay value in expriment script)      
        n_delay_samples=preprocess_config.stim_delay_time*EEG.srate; 
        disp('**** NOTE THA THIS FUNCTION ONLY MAKES SENSE IF NOT USING CLICKS AND SPEECH DELAY TIME WAS ACTUALLY ADDED')

        fprintf('adding %0.2fs /%d samples\n',preprocess_config.stim_delay_time,n_delay_samples)
    
        % TODO: ASK AARON HOW TO DO THIS WITHOUT LOOPS FOR THE LOVE OF
        % ALL THAT IS HOLY
        % RE: above - I did and his idea didn't work but there must be
        % another way....
        for trial_indx=find([EEG.event.type]<=preprocess_config.n_trials)
            EEG.event(trial_indx).latency=EEG.event(trial_indx).latency+n_delay_samples;
        end
    end
    if ~isempty(preprocess_config.stim_delay_time)
        % note: this should definitely happen before padding calculations
        EEG=add_speech_delay(EEG,preprocess_config);
    end

    % check if last condition is slow, otherwise need to pad EEG.data
    % so pop_epoch works (last trial needs to be within epoch_dur
    % boundary)
    function dur_last_trial=last_trial_dur(EEG)
        % assumes EEG.event types only pertain to start of trials
        % also assumes EEG.times in ms....
        end_time=EEG.times(end)*1e-3;
        final_trial_start_time=(EEG.event(end).latency-1)/EEG.srate;
        dur_last_trial=end_time-final_trial_start_time;
    end

    if last_trial_dur(EEG)<preprocess_config.epoch_dur       
        ns_pad=ceil((preprocess_config.epoch_dur-last_trial_dur(EEG))*EEG.srate);
        % append zeros to end of dataset so pop_epoch can work
        EEG.data=[EEG.data, zeros(preprocess_config.nchan,ns_pad)];
    end

    disp('copying eeg pre-epoching to compate plots')
    pre_EEG=EEG; % debugging coppy
    EEG = pop_epoch(EEG, ...
        num2cell(preprocessed_eeg.trials),[0 preprocess_config.epoch_dur]);

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


