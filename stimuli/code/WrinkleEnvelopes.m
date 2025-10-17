clear;
clc;
global boxdir_mine boxdir_lab

pilot=true; % fetch reg/irreg pilot data
overwrite = 0;
% TODO: VERIFY CORRECT STIMFOLDER FOR PILOT DATA
%TODO: RENAME OLD ENVELOPES FNMS TO EXCLUDE WRINKLE IN FNM - ADD VARIABLE
%CALLED STIMGROUP TO THEM + UPDATE TRF_ANALYSIS_SCRIPT PATHS TO REFLECT
%CHANGE
%NOTE: moving final stimset here once happy with end result - use separate
%for pilot
fs = 128; %for trfs
% fs=441; % for coherence analysis
stimgroup = 'wrinkle';


if pilot
    % use for reg/irreg (pilot)
    stimscale=[1 1 1];
    regularity=[-1 0 1];
    ntrials = 75;
    outputFile=sprintf('%s/stimuli/%s/RegIrregPilotEnvelopes%dhz.mat', ...
        boxdir_mine,stimgroup,fs);
    pilot_stimfolder = 'rule11_seg_textgrid_4p545Hz_0_0';%note: there are two folders here... but presuming we used same folder name (rule 11 with parmsm)
else
    % use for fast/slow
    stimscale = [2/3 1 3/2];
    regularity=[0 0 0];
    ntrials = 120;
    outputFile=sprintf('%s/stimuli/%s/Envelopes%dhz.mat',boxdir_mine,stimgroup,fs);
    
end

pilot_stimfolder = 'rule11_seg_textgrid_4p545Hz_0_0';
stimfolder = sprintf('%s/stimuli/',boxdir_lab);
m=[stimscale;regularity];


[env,sgram]=deal(cell(length(stimscale),ntrials));
%%
for cc = 3:length(m) % note: change start of loop to cc=1 after script ends
    for tt = 1:ntrials
        fprintf('**********************\n')
        fprintf('Speed = %0.2g, regularity = %0.2g, trial %d\n',stimscale(cc),regularity(cc),tt)
        fprintf('**********************\n')
        switch regularity(cc)
            case 0
                if stimscale(cc)==1
                    audiofile = sprintf('%s%s/og/%s%0.3d.wav',stimfolder, ...
                        stimgroup,stimgroup,tt);
                else
                    audiofile = sprintf('%s%s/%0.2f/%s%0.3d.wav', ...
                        stimfolder,stimgroup,stimscale(cc),stimgroup,tt);
                end
            case -1
                % TODO: verify compressy is -1 in all other scripts so
                % ordering is correct in output
                audiofile=sprintf(['%s/stimuli/%s/stretchy_compressy_temp/' ...
                    'compressy_reg/%s/%s%0.3d.wav'],boxdir_mine,stimgroup, ...
                    pilot_stimfolder,stimgroup,tt);
            case 1
                audiofile=sprintf(['%s/stimuli/%s/stretchy_compressy_temp/' ...
                    'stretchy_irreg/%s/%s%0.3d.wav'],boxdir_mine,stimgroup, ...
                    pilot_stimfolder,stimgroup,tt);
            otherwise
                warning('invalid regularity %0.2g',regularity(cc))
        end
        
        [env{cc,tt}, sgram{cc,tt}] = extractGCEnvelope(audiofile,fs);
    end
end
save(outputFile,'env','sgram','fs','stimgroup')