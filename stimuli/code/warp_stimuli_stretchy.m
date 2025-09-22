% use this scriptf - implements stretchy/compressy rules

%TODO: use-preloaded envelopes to speed up computation time
clear;
clc;
global boxdir_lab
global boxdir_mine
inputStimfolder = sprintf('%s/stimuli/',boxdir_lab);
outputStimfolder=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/',boxdir_mine);
%need TSM toolbox

% addpath(sprintf('%s/Box/my box/LALOR LAB/matlab-toolboxes/MATLAB_TSM-Toolbox_2.01/MATLAB_TSM-Toolbox_2.01',userProfile))
% save output to my personal box location - avoid mixing files with shared
% drive
warp_rules=[11];
% make freqs -1, 0, or 1 to use mean median or mode
% upper/lower quartiles: 6.358/3.940 Hz (estimated using truncated gamma
% pdf fit to bark-derived peakRates)

% center_freqs=[6.358,3.940, 0];
%updated quartiles to reflect empirical distribution values
% center_freqs=[4.12960014982676,	0,	8.21688093907211];
% center_freqs=[-1 0 1] -> [lquartile median uquartile]
% -1 -> lquartile, 0->median, 1-> uquartile
center_freqs=4;
% center_freqs=[6.358,3.940, 0]; %NOTE: we assuming only 3 values given at a time MAXIMUM (one of which is median...)
% and all values greater than 1 since 1 means mode....
conditions=[2]; % 1-> irreg (stretchy) 2-> reg (compessy)

cond_nms={'stretchy_irreg','compressy_reg'};
% center_freqs=logspace(0.4,.95,10);
overwrite = 1;
% stimgroup = {'leagues','oldman','wrinkle'};
stimgroup = {'wrinkle'};
% stimspeed = [0.5 2/3 3/2 2];
% seems we can bypass speed warping by feeding empty array:
stimspeed=[];


for warp_rule_num=warp_rules
    fprintf('warp_rule_num: %d\n',warp_rule_num)
    for ff = 1:length(center_freqs)
        center_f=center_freqs(ff);
        switch center_f
            case -1
                % center_str='mean';
                center_str='lquartile';
            case 0
                %TODO: redefine median based on correct value across
                %stimuli....
                center_str='median';
            case 1
                % center_str='mode';
                center_str='uquartile';
            otherwise

                center_str=sprintf('%0.3fHz',center_f);
        end
        
    for gg = 1:length(stimgroup)
        fprintf('processing stimulus: %s\n',stimgroup{gg})
    
        d = dir(sprintf('%s%s/og/*.wav',inputStimfolder,stimgroup{gg}));
        for dd = 1:length(d)
            fprintf('file: %s \n',d(dd).name)
            audiofile = sprintf('%s%s/og/%s',inputStimfolder,stimgroup{gg},d(dd).name);
            [audio,wav_fs]=audioread(audiofile);
            
            for cc=conditions
                cond_nm=cond_nms{cc};
                warp_config=[];
                warp_config.rule_num=warp_rule_num;
                switch cc
                    case 1 % irreg
                        warp_config.k=1;
                    case 2 % reg
                        warp_config.k=-1;
                end
                warp_config.center=center_f;
                warp_config.env_method='bark';
                % warp_config.env_method='oganian';
                warp_config.env_thresh_std=1e-3;
                warp_config.jitter=0.05;
                warp_config.sil_tol=1;
                % in Hz, rate which is considered too fast to count as new syllable from input distribution

                warp_config.hard_cutoff_hz=14; 
                %note: added this new param after noticing that syllables
                %sometimes occur above 10 Hz so maybe shouldn't lowpass the
                %envelope blanketly at 10 hz...
                warp_config.env_lpf=14;
                % warp_config.env_derivative_noise_tol=0;
                % minimal as to eliminate spurious peaks during silences
                warp_config.min_pkrt_height=5e-5;
                % warp_config.prom_thresh=0.2;
                % warp_config.width_thresh=0.01;
                % warp_config.area_thresh=2.40e-6;
                warp_config.rng=1; %note: probably want this to be false once we're confident manual filter works
                warp_config.manual_filter=1;
                

                outputDirTemp=sprintf('%s%s/rule%d_seg_%s_%s_%d_%d/', ...
                        outputStimfolder,cond_nm,warp_config.rule_num, ...
                        warp_config.env_method,center_str,warp_config.rng, ...
                        warp_config.manual_filter);
                audioOutputFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name);
                audioParamsFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name(1:end-4));
    
                if ~exist(audioOutputFileTemp,'file') || overwrite
                    fprintf('%s - f: %0.3f\n',cond_nm,center_f)
                    [warped_audio,S] = stretchyWrinkle(audio,wav_fs,warp_config); 
    
                    if ~exist(outputDirTemp,'dir')
                        mkdir(outputDirTemp)
                    end
                    audiowrite(audioOutputFileTemp,warped_audio,wav_fs);
                    save(audioParamsFileTemp,"S");
                    clear audioOutputFileTemp audioParamsFileTemp warped_audio S
                end
            end
        
    
            fprintf('\n')
        end
    end
    end
end
