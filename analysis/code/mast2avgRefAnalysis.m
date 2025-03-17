% for this script just load the mast and avg ref versions, compare the
% r-values in each condition
clear
clc
dependencies_path=('../../dependencies/');
addpath(genpath(dependencies_path));
%TODO: add envelope/eeg normalization

subj = 5;
%preprocessing params
do_correction=true; % add one sec to event latencies before epoching to account for speech delay in noisy stims
bpfilter = [1 15];
refs = {'avg', 'mast'};
fs = 128;
% nchan = 128;
% refI = nchan+(1:2);
% opts = {'channels',1:(nchan+2),'importannot','off','ref',refI};
% if ismember(subj, [5 8])
%     interpBadChans=true;
% else
%     interpBadChans=false; % only seems necessary for subj 5..?
% end
% % trf params
% zscore_envs=false;
% norm_envs=true;
% zscore_eeg=true;
% separate_conditions=true;
% lam=[1e3]; %vector of lambda values to use in mtrf crossval
% % lam=10.^[-3:8]; %idk how to choose a good range here
% mtmin=-500; % for mTRFtrain model
% mtmax=800; % for mTRFtrain model
% if separate_conditions
conditions_dir='sep_conditions/';
% else
%     conditions_dir='all_conditions/';
% end
% if do_correction
corrdir='corrected/';
% else
%     corrdir='notcorr/';
% end
% 
% if ~isscalar(lam)
%     cvdir='cv/';
% else
    % cvdir='';
    % end
cvdir='';
user_profile=getenv('USERPROFILE');
% envelopesFile=sprintf('../stimuli/WrinkleEnvelopes%dhz.mat',fs);
datafolder = sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/data/',user_profile);
behfile = sprintf('%ss%0.2d_WarpedSpeech.mat',datafolder,subj);
bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf',datafolder,subj);
for refIdx=1:numel(refs)
    ref=refs{refIdx};
    matfolder = sprintf('%smat/%g-%g_%s-ref_%dHz/%s',datafolder,bpfilter(1),bpfilter(2),ref,fs,corrdir);
    
    % matfile = sprintf('%swarpedSpeech_s%0.2d.mat',matfolder,subj);
    

    nulldistribution_file=sprintf('%s%s%snulldistribution_s%0.2d.mat',matfolder,conditions_dir,cvdir,subj);
    fprintf('loading results from %s\n',nulldistribution_file)
    results(refIdx)=load(nulldistribution_file,'model','model_lam','r_obs');
end
%% plot trfs side by side
chns=[85];
conditions=1:3;
ccNms={'fast','og','slow'};
figure
for refIdx=1:numel(refs)
    ref=refs{refIdx};
    subplot(2,1,refIdx);
    hold on
    for cc=conditions
        tempModel=results(1,refIdx).model{cc};
        tempW=tempModel.w(:,:,chns);
        tempT=tempModel.t;

        plot(tempT,tempW,'DisplayName',ccNms{cc})
        clear tempW tempT tempModel
        
    end
    legend()
    xlabel('Time (ms)')
    title(ref)
    hold off
end

%% plot trfs difference
figure
for cc=conditions
    ref=refs{refIdx};
    subplot(3,1,cc);
    hold on
    for refIdx=1:numel(refs)
        tempModel=results(refIdx).model{cc};
        tempW(refIdx,:)=tempModel.w(:,:,chns);
        tempT(refIdx,:)=tempModel.t;
    end
    tempDiff=tempW(1,:)-tempW(2,:);
    plot(tempT(1,:),tempDiff,'DisplayName',ccNms{cc})
        clear tempW tempT tempModel tempDiff
    legend()
    xlabel('Time (ms)')
    
    hold off
end
sgtitle(sprintf('%s-%s model weights',refs{1},refs{2} ))

%% plot bias difference
figure
for cc=conditions
    subplot(3,1,cc);
    hold on
    for refIdx=1:numel(refs)
        tempModel=results(refIdx).model{cc};
        tempB(refIdx,:)=tempModel.b;

    end
    tempDiff=tempB(1,:)-tempB(2,:);
    plot(1:128,tempDiff,'DisplayName',ccNms{cc})
        clear tempB tempT tempModel tempDiff
    xlabel('electrodes')
    legend()
    
    hold off
end
sgtitle(sprintf('bias difference (%s-%s)',refs{1},refs{2} ))

%% plot r_obs(avgref) vs r_obs(mastref)
figure
for cc=conditions
    subplot(3,1,cc);
    hold on
    for refIdx=1:numel(refs)
        tempRobs(refIdx,:)=results(refIdx).r_obs(cc,:);
        % tempB(refIdx,:)=tempModel.b;
        % tempT(refIdx,:)=tempModel.t;
    end
    maxR=max(tempRobs,[],'all');
    plot(tempRobs(1,:),tempRobs(2,:),'DisplayName',ccNms{cc})
    plot([0 maxR], [0 maxR],'r--')
        % clear 
    xlabel(sprintf('%s r-vals',refs{1}))
    ylabel(sprintf('%s r-vals',refs{2}))
    legend()
    xlim([0 maxR])
    ylim([0 maxR])
    hold off
end

% sgtitle(sprintf('bias difference (%s-%s)',refs{1},refs{2} ))









% 
% 
% % for finding chanlocs file:
% % boxfolder=sprintf('%s/Box/Lalor Lab Box/Code library/EEGpreprocessing/',user_profile);
% % Segment parameters
% % seems liike rec_dur is in samples... 
% seg_dur = 10;
% rec_dur = 96; 
% segments = (1:seg_dur*fs:rec_dur*fs)';
% segments(:,2) = segments(:,1)+seg_dur*fs-1;
% segments(any(segments>rec_dur*fs,2),:) = [];
% n_trials=75;
% 
% if exist(nulldistribution_file,'file')
%     load(nulldistribution_file)
% else
% 
%     if exist(matfile,'file')
%         disp('existing preprocessed mat file found, loading from mat.')
%         load(matfile,'stim','resp','events','cond','fs')
%     else
% 
% 
%         EEG = pop_biosig(bdffile,opts{:});
%         % remove mastoids
%         EEG = pop_select(EEG,'nochannel',nchan+(1:2));
%         chanlocsfile = sprintf('%schanlocs.xyz',boxfolder);
%         EEG = pop_chanedit(EEG,'load',{chanlocsfile,'filetype','xyz'});
%         EEG.urchanlocs = EEG.chanlocs;
% 
%         % resample
%         EEG = pop_resample(EEG,fs);
%         if interpBadChans
%             % note that resp will have the interpolated channels but
%             % original bdf will not
%             disp('interping bad channels')
%             badChans=findBadChans(EEG.data'); %how to choose thresholds?
%             EEG=eeg_interp(EEG,badChans);
%         end
%         % filter to frequency band of interest
%         hd = getHPFilt(EEG.srate,bpfilter(1));
%         EEG.data = filtfilthd(hd,EEG.data')';
% 
% 
%         hd = getLPFilt(EEG.srate,bpfilter(2));
%         EEG.data = filtfilthd(hd,EEG.data')';
% 
%         % Epoch
%         events = [EEG.event.type];
%         % why did aaron choose 100 in particular rather than n_trials to
%         % begin with?
%         events(events>n_trials+1) = [];
%         % remove repeated trials from EEG structure first, if any
%         false_start=check_restart(events);
%         if false_start
%             %note: i think this only works if there is only a single false
%             %start
%             % if multiple restarts, grab the final one
%             start_iis=[0 diff(events)];
%             last_start_ii=find(start_iis<1,1,'last');
%             max_repeated_trial_type=events(last_start_ii-1);
%             events=events(last_start_ii:end);
%             % remove false start trials from EEG structure so epoch won't
%             % return multiple trials for those
%             false_start_end_ii=find([EEG.event.type]==max_repeated_trial_type,1,'first');
%             % stupid loop because idk how to assign multiple struct array
%             % vals otherwise
%             for rm_trial=1:false_start_end_ii
%                 EEG.event(rm_trial).type=3000;
%             end
% 
%             % below code only works if events only has trial numbers, which
%             % EEG.event does not...
%             % false_start=check_restart([EEG.event.type]);
%             % if false_start
%             %     error('that didnt work')
%             % end
%         end
%         if ~exist(matfolder,'dir')
%             mkdir(matfolder)
%         end
%         load(behfile,'m')
%         cond = round(m(:,1),2);
%         cond(cond>1) = 3;
%         cond(cond==1) = 2;
%         cond(cond<1) = 1;
%         % check if last condition is slow, otherwise need to pad EEG.data
%         % so pop_epoch works (last trial needs to be within rec_dur
%         % boundary)
%         if cond(end)~=3
%             % slow condition is fine, others need padding
% 
%             ns_pad=floor((rec_dur)*EEG.srate);
%             EEG.data=[EEG.data, zeros(nchan,ns_pad)];
%         end
%         if do_correction
%             % shift trial starts by 1 second (speech_delay value in AAA
%             % code)
%             % note this will 
%             disp('correcting latencies for speech delay')
% 
%             delay_time=1; %in seconds
%             ns_correction=delay_time*EEG.srate; % TODO: double check if off by one
%             % note: pop_editeventfield does not work as expected...s events
%             % have out-of-bounds latencies and removes them even though
%             % that is false...
%             % onset_indx=[EEG.event.type]<=n_trials;
%             % onsets=[EEG.event.latency];
%             % onsets(onset_indx)=onsets(onset_indx)+ns_correction;
%             % EEG=pop_editeventfield( EEG, 'latency',onsets);
%             % TODO: ASK AARON HOW TO DO THIS WITHOUT LOOPS FOR THE LOVE OF
%             % ALL THAT IS HOLY
%             for trial_indx=find([EEG.event.type]<=n_trials)
%                 EEG.event(trial_indx).latency=EEG.event(trial_indx).latency+ns_correction;
%             end
%         end
%         EEG = pop_epoch(EEG,mat2cell(events,1,ones(1,numel(events))),[0 rec_dur]);
% 
%         resp = cell(1,size(EEG.data,3));
%         for tt = 1:size(EEG.data,3)
%             resp{1,tt} = EEG.data(:,:,tt)';
%         end
% 
% 
% 
%         load(envelopesFile,'env')
%         fs_stim=load(envelopesFile,'fs');
%         fs_stim=fs_stim.fs;
%         % check wav fs matches analysis fs
%         if fs_stim ~= fs
%             error('stim has wrong fs.')
%         end
%         stim = env(cond,1:n_trials);
%         stim = stim(logical(eye(size(stim))));
% 
% 
%         cond = cond(events,1);
%         stim = stim(events,1)';
% 
%         % segmenting
%         % if seg
%         %     resp_seg = cell(nStim*size(segments,1),length(cond1));
%         %     stim_seg = cell(nStim*size(segments,1),length(cond1),size(stimAll,2));
%         %
%         %     tt=0;
%         %     for tr = 1:size(stim,1)
%         %         for sg = 1:size(segments,1)
%         %             tt = tt + 1;
%         %             for c1 = 1:size(stim,2)
%         %                 resp_seg{1,tt} = resp{tr}(segments(sg,1):segments(sg,2),:);
%         %                 stim_seg{1,tt} = stim{tr}(segments(sg,1):segments(sg,2),:);
%         %             end
%         %         end
%         %     end
%         %     if seg==1
%         %         resp = resp_seg; clear resp_seg
%         %         stim = stim_seg; clear stim_seg
%         %     end
%         % end
%         for tt = 1:size(stim,2)
%             resp{tt} = resp{tt}(1:size(stim{tt},1),:);
%         end
% 
%         save(matfile,'stim','resp','events','cond','fs','interpBadChans')
%     end
%     if zscore_envs
%         % NOTE: need to check if they've already been z-scored before doing
%         % this.... or not because it will always load from the mat file?
%         if norm_envs
%             error('dont do both normalization and z-scoring on envelopes')
%         end
%         disp('z-scoring envelopes')
%         load(envelopesFile,'mu','sigma');
%         stim=cellfun(@(x) (x-mu)/sigma, stim,'UniformOutput',false);
%     end
%     if norm_envs
%         disp('normalizing envelopes')
%         load(envelopesFile,'sigma');
%         stim=cellfun(@(x) x/sigma, stim,'UniformOutput',false);
%     end
% 
%     if zscore_eeg
%         disp('z-scoring eeg')
%         % concatenate all trials
%         resp_cat=cat(1,resp{:,:});
%         % z-score all the channels together
%         eeg_mu=mean(resp_cat,'all');
%         eeg_sigma=std(resp_cat,0,'all');
%         resp=cellfun(@(x)(x-eeg_mu)./eeg_sigma,resp,'UniformOutput',false);
%         clear resp_cat
%     end
% 
%     if separate_conditions
%         conditions=unique(cond)';
%         % assuming cells preserve order, can map back to original 
%         stats_obs=cell(numel(conditions),1);
%         for cc=conditions
%             cc_mask=cond==cc;
%             stats_obs{cc}=mTRFcrossval(stim(cc_mask),resp(cc_mask),fs,1,0,400,lam,'Verbose',0);
%         end
% 
%     else
%         stats_obs = mTRFcrossval(stim,resp,fs,1,0,400,lam,'Verbose',0);
%     end
%     % get best lambda 
%     %NOTE: not tested yet
%     if isscalar(lam)
%         model_lam=lam;
%         do_nulltest=true;
%     else
%         % average out trials
%         r_avg_trials=squeeze(mean(stats_obs.r,1));
%         % get max across electrodes for each lambda
%         r_max_electrodes=squeeze(max(r_avg_trials,[],2));
%         % get index of max r-value 
%         [~,best_lam_idx]=max(r_max_electrodes);
%         model_lam=lam(best_lam_idx);
%         do_nulltest=true;
% 
% 
%     end
%     fprintf('best lambda value: %0.1g\n',model_lam)
%     if separate_conditions
%         disp('evaluating trf models separately per condition')
%         model=cell(numel(conditions),1);
%         for cc=conditions
%             fprintf('TRF for condition %d...',cc)
%             cc_mask=cond==cc;
%             model{cc}=mTRFtrain(stim(cc_mask),resp(cc_mask),fs,1,mtmin,mtmax,model_lam,'Verbose',0);
%         end
%     else
%         model = mTRFtrain(stim,resp,fs,1,mtmin,mtmax,model_lam,'Verbose',0);
%     end
% 
%     if do_nulltest
%         msg = 0;
%         for ii = 1:1000
%             if ~mod(ii,50)
%                 fprintf(repmat('\b',msg,1))
%                 msg = fprintf('iteration #%0.4d',ii);
%             end
%             resp_shuf = resp;
%             for cc = 1:3
%                 I = find(cond==cc);
%                 I2 = I(randperm(length(I)));
%                 resp_shuf(I2) = resp(I);
%             end
% 
%             if separate_conditions
% 
%                 stats_null=cell(3,1);
%                 % r_null=cell(3,1);
%                 for cc=conditions
%                     % fprintf('null TRF for condition %d...\n',cc)
%                     cc_mask=cond==cc;
%                     r_obs(cc,:) = squeeze(mean(stats_obs{cc}.r,1));
%                     stats_null{cc}=mTRFcrossval(stim(cc_mask),resp_shuf(cc_mask),fs,1,0,400,model_lam,'Verbose',0);
%                     r_null(cc,ii,:) = squeeze(mean(stats_null{cc}.r,1));
%                 end
%             else
%                 if isscalar(lam)
%                     obs_rvals=stats_obs.r;
%                 else
%                     %todo:verify this is valid? don't care about actual
%                     %result anyhow so just leaving as is so that script
%                     %fucking runs
%                     obs_rvals=squeeze(stats_obs.r(:,best_lam_idx,:));
%                 end
%                 r_obs(1,:) = squeeze(mean(obs_rvals,1));
%                 stats_null = mTRFcrossval(stim,resp_shuf,fs,1,0,400,model_lam,'Verbose',0);
%                 r_null(ii,:) = squeeze(mean(stats_null.r,1));
%             end
%         end
%     else
%         disp('not doing null test, only do it when single lambda provided bc too annoyed with changing the code to account for cv case')
%     end
%     nulldir=fileparts(nulldistribution_file);
%     if ~exist(nulldir,'dir'), mkdir(nulldir);end
%     save(nulldistribution_file)
% end
% 
% 
% plot_ch=85;
% 
% % if do_correction
% %     corr_title_str='with';
% % else
% %     corr_title_str='without';
% % end
% if separate_conditions
%     for cc=conditions
%         figure
%         hist(r_null(cc,:,plot_ch))
%         title(sprintf('subj %d, cond %d, chn %d null distribution - \\lambda %.3g',subj,cc,plot_ch,model_lam))
%         figure
%         ecdf(r_null(cc,:,plot_ch))
%         hold on
%         plot(repmat(r_obs(cc,plot_ch),1,2),ylim,'r')
%         title(sprintf('subj %d, cond %d, chn %d permutation test - \\lambda %.3g',subj,cc,plot_ch,model_lam))
%     end
% 
% else
%     figure
%     hist(r_null(:,plot_ch))
%     title(sprintf('subj %d, chn %d null distribution - \\lambda %.3g',subj,plot_ch,model_lam))
%     figure
%     ecdf(r_null(:,plot_ch))
%     hold on
%     plot(repmat(r_obs(plot_ch),1,2),ylim,'r')
%     title(sprintf('subj %d, chn %d permutation test - \\lambda %.3g',subj,plot_ch,model_lam))
% end
% 
% function restart_bool=check_restart(event_trials)
% % helper function to check if there are excess trials from restarting
% % should work even if first trial is not trial 1
% if any(diff(event_trials)<1)
%     restart_bool=true;
% else
%     restart_bool=false;
% end
% end