% getPeakRate
% TODO: make code do the stuff stated below:
% extract peakRate using same algo as warp stimuli
% checks that timestamps generated match those for warped stimuli
% extracts peakrate from already warped stimuli (rather than computing from
% stimulus like getPeakRate.m)
% first col in output mats is og condition
clear, clc
global boxdir_mine
out_dir=[boxdir_mine '/stimuli/peakRate/'];

soundchn=1;
% conditions={'compressy_reg','stretchy_irreg'};
% conditions={'rand','og'};
% 0 -> og 1 -> regular -1-> irregular  2/3-> slow 3/2-> fast

% conditions=[0,2/3,3/2];
% conditions=[-1 1];
overwrite=0;

% get just s-mat files that have timing of peakrate events from warping
% script run
get_smat=false;
% "C:\Users\apalaci6.URMC-SH\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_blanket_normalized_durations"
cond_nm='rule8_seg_bark_median_speedNormalizedDurations';
cc_source_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/stretchy_irreg/%s/',boxdir_mine,cond_nm);
% % og in separate folder: NOTE - I think we decided this was not the issue
% and we ddin't need the shorter files necessarily?
% cond_nm='og';
% cc_source_dir=sprintf('%s/stimuli/wrinkle_wClicks/%s/',boxdir_mine,cond_nm);
% global boxdir_lab
% cc_source_dir=sprintf('%s/stimuli/wrinkle/%s/',boxdir_lab,cond_nm);


out_flpth=sprintf('%s%s.mat',out_dir,cond_nm);

% for cc=1:numel(conditions)
peakRate=struct('times',[],'amplitudes',[],'prominence',[],'peakwidth',[],'peakwidth2',[]);
% cond=conditions(cc);

% [cc_source_dir,cond_nm]=get_source_dir_cond(cond);
fprintf('processing %s condition...\n',cond_nm)
% out_flpth=sprintf('%s%s.mat',out_dir,cond_nm);
if ~exist(out_flpth,'file') || overwrite
    if exist(out_flpth,'file')
        fprintf('%s exists already, overwriting...\n',out_flpth)
    end
    D=dir([cc_source_dir '*.wav']);
    for nn=1:numel(D)
        fprintf('%d/%d\n',nn,numel(D))
        wavpath=[cc_source_dir D(nn).name];
        [wf,fs]=audioread(wavpath);
        env=bark_env(wf(:,soundchn),fs,fs);
        % [peakTs, peakVals]=get_peakRate(env,fs);
        [peakRate(nn).times, peakRate(nn).amplitudes, peakRate(nn).prominence,... 
            peakRate(nn).peakwidth, peakRate(nn).peakwidth2]=get_peakRate(env,fs);
        % pkIs=round(peakTs.*fs);
        % peakRate(nn).times=peakTs;
    end
    fprintf('saving output...\n')
    save(out_flpth,"peakRate","cc_source_dir","fs")
else
    fprintf('file %s exists already, skipping\n',out_flpth)
end
clear out_flpth D peakRate cc_source_dir cond_nm
% end

%% Verify results NOTE: better do everything below on separate script "plot_peakrate_dist"
%% quick and dirty time domain plot
plot_time_domain=false;
if plot_time_domain
    n_display_stim=40;
    scale_multiplier=1e4;
    for cc=1:numel(conditions)
        cond=conditions(cc);
        
        [cc_source_dir,cond_nm]=get_source_dir_cond(cond);
        fprintf('%s...\n',cond_nm)
        D=dir([cc_source_dir '*.wav']);
        sample_path=[cc_source_dir D(n_display_stim).name];
        [sample_wf,fs]=audioread(sample_path);
        sample_env=bark_env(sample_wf(:,soundchn),fs,fs);
        [peakTs,peakVals]=get_peakRate(sample_env,fs);
        figure, plot((0:(numel(sample_env)-1))./fs,sample_env), hold on
        stem(peakTs,peakVals.*scale_multiplier)
        title(sprintf('%s - stim # %d',cond_nm, n_display_stim))
    end
end


% if get_smat
% % set threshold for peaks to ignore based on smoothed envelope value
% nwavs=120; % per condition
% nconditions=numel(conditions);
% %TODO: OOPS forgot to save the actual peakrate values.... re-run and save
% %those for analysis too
% % NOTE: actually this might be impossible since we didn't save them just
% % the s matrix hehehehe...
% % peakRate=cell(nconditions+1,nwavs);
% peakRateIntervals=cell(nconditions+1,nwavs);
% peakRateFreqs=cell(nconditions+1,nwavs);
% % peakR
% 
% 
% 
% if ~exist(peakrateFileOut,'file')
%     for cc=1:nconditions
%         [wavspath,D]=getFilePaths(stimulifolder,conditions(cc));
% 
%         for windx=1:length(D)
%             fprintf('wav %d...\n',windx)
%             s_flpth=sprintf('%s/%s.mat',wavspath,D(windx).name(1:end-4));
%             load(s_flpth,'s_temp');
%             %TODO: get the og jsut on the first iteration (make a cell for
%             %it) then also for the two warp conditions (and calculate
%             %rates)
%             % assume all audio at same fs - just read first sample of first
%             if cc==1 && windx==1
%                 aud_flpth=sprintf('%s/%s',wavspath,D(windx).name);
%                 [~,fs]=audioread(aud_flpth, [1 2]);                
%             end
%             if cc==1
%                 peakRateIntervals{cc,windx}=get_pr_intervals(s_temp(:,1),fs);
%                 peakRateFreqs{cc,windx}=1./[peakRateIntervals{cc,windx}];
%             end
% 
%             peakRateIntervals{cc+1,windx}=get_pr_intervals(s_temp(:,2),fs);
%             peakRateFreqs{cc+1,windx}=1./[peakRateIntervals{cc+1,windx}];
% 
% 
% 
%         end
%         clear D wavspath windx
%     end
%     save(peakrateFileOut)
% else
%     load(peakrateFileOut)
% end
% disp('done')
% end
%% helpers
function [cc_source_dir,cond_nm]=get_source_dir_cond(cond)
global boxdir_mine
stimset='wrinkle';
stimulifolder=sprintf('./%s_wClicks/',stimset);

% different location for pilot stimuli - subject to change
% "C:\Users\apalaci6.URMC-SH\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_blanket_normalized_durations"
pilot.stimulifolder=sprintf('%s/%s/stretchy_compressy_temp/stretchy_irreg/rule2_seg_bark_median_blanket_normalized_durations/',stimset);
pilot.irreg_rule='stretchy_irreg/rule2_seg_bark_median/';
pilot.reg_rule='compressy_reg/rule5_seg_bark_median/';
    if mod(cond,1)==0
        switch sign(cond)   
            case 0
                %og
                cc_source_dir=sprintf('%sog/',stimulifolder);
                cond_nm='og';
            case -1
                cc_source_dir=sprintf('%s%s',pilot.stimulifolder,pilot.irreg_rule);
                cond_nm='irreg';
            case 1
                cc_source_dir=sprintf('%s%s',pilot.stimulifolder,pilot.reg_rule);
                cond_nm='reg';
        end
    else
        cc_source_dir=sprintf('%s%0.2f/',stimulifolder,cond);
        if cond>1
            cond_nm='fast';
        else
            cond_nm='slow';
        end
    end
end
function intervals=get_pr_intervals(s,fs)
%TODO: account for long pauses here...
% sil_tol (1,1) double = 0.75; - is this what was actually used in script?
% given s matrix with TSM algo anchorpoints based on peakRate, calculate
% time intervals (in s) between peaks
% assume a column vector given - single audiofile
%ignore first and last inices since meaningless
s([1,end])=[];
intervals=diff(s)./fs;
end
function [wavspath,D]=getFilePaths(stimulifolder,currentCondition)
    if isnumeric(currentCondition)
        switch round(currentCondition,2)
            case 1
                wavspath=sprintf('%s/og',stimulifolder);
            case {.67, 1.5}
                wavspath=sprintf('%s/%0.2f',stimulifolder,currentCondition);
        end
        fprintf('fetching wav file paths for condition: %0.2f\n', currentCondition)
    elseif iscell(currentCondition)
        wavspath=sprintf('%s/%s',stimulifolder,currentCondition{:});
        fprintf('fetching wav file paths for condition: %s\n',currentCondition{:})
    D=dir(sprintf('%s/wrinkle*.wav',wavspath));
    end
end