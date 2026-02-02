%general notes:
% this script is confusing because it originally searched boxdir_lab for 
% source audiofiles, but the foldertree structure is somewhat different on
% there now relative to our temp folder where we look for newly generated
% stimuli originally... probably will save time by simply ignoring the
% boxdir_lab more broadly and simply using it to preserve/share final
% results manually.... as a general rule not just for this script
clear;
clc;
global boxdir_mine boxdir_lab
config.experiment='reg/irreg';
config.pilot=false;
config.search_temp=true;
%NOTE: folder name below might cause problems later if we change to
%using a rule besides 14 because we added a max interval descriptor to the
%end of the temp folder name in irreg case but not reg - and we're
%addressing it by simply indexing the character vector to keep first 31
%characters in reg case. a more general solution would be nice here....
config.which_temp='rule14_seg_textgrid_4p545Hz_0_0_1000ms_max';
config.overwrite = 0;
% note: use boxdir_mine for temp stimuli, boxdir_lab for final stimuli used
% structure in both should match but I haven't been doing a goog job of
% updating lab box folders
stimfolder = sprintf('%s/stimuli/',boxdir_mine);
fs = 128; 
stimgroup = 'wrinkle';
click_chn=2;
switch config.experiment
    case 'reg/irreg'
            stimscale=[1 1 1];
            regularity=[-1 0 1];
            ntrials = 75;
        if config.pilot
            outputFile=sprintf('%s/stimuli/%s/RegIrregPilotEnvelopes%dhz.mat', ...
                boxdir_mine,stimgroup,fs);
            
        else
            outputFile=sprintf('%s/stimuli/%s/regIrregEnvelopes%dhz.mat', ...
                boxdir_mine,stimgroup,fs);
        end
        if config.search_temp
            regfolder=['stretchy_compressy_temp/compressy_reg/',  ...
                config.which_temp(1:31)];
            irregfolder=['stretchy_compressy_temp/stretchy_irreg/', ...
                config.which_temp];
        else
            % store only the final product used in experiment here
            regfolder='reg';
            irregfolder='rand';
        end

    case 'fast/slow'
        stimscale = [2/3 1 3/2];
        regularity=[0 0 0];
        ntrials = 120;
        if config.pilot
            error('naw dawg')
        else
            outputFile=sprintf('%s/stimuli/%s/fastSlowEnvelopes%dhz.mat',boxdir_mine,stimgroup,fs);
        end
        if config.search_temp
            error('naw, dawg');
        end
end

stim_conditions=[stimscale;regularity];


[env,sgram]=deal(cell(length(stimscale),ntrials));
%%
for cc = 3:length(stim_conditions) % note: change start of loop to cc=1 after script ends
    for tt = 1:ntrials
        fprintf('**********************\n')
        fprintf('Speed = %0.2g, regularity = %0.2g, trial %d\n',stimscale(cc),regularity(cc),tt)
        fprintf('**********************\n')
        switch regularity(cc)
            case 0
                % fast-slow or og
                %note: looking in folder w clicks now since restricting to \
                % my boxdrive to simplify things... doesnt seem like they
                % have clicks though
                if stimscale(cc)==1
                    audiofile = sprintf('%s%s/og/%s%0.3d.wav',stimfolder, ...
                        [stimgroup '_wClicks'],stimgroup,tt);
                else
                    audiofile = sprintf('%s%s/%0.2f/%s%0.3d.wav', ...
                        stimfolder,[stimgroup '_wClicks'],stimscale(cc),stimgroup,tt);
                end
            case 1
                % irreg
                audiofile=sprintf('%s%s/%s/%s%0.3d.wav',stimfolder, ...
                    stimgroup,irregfolder,stimgroup,tt);
            case -1
                % reg
                audiofile=sprintf('%s%s/%s/%s%0.3d.wav',stimfolder, ...
                    stimgroup,regfolder,stimgroup,tt);
            otherwise
                warning('invalid regularity %0.2g',regularity(cc))
        end
        %TODO: CHECK IF CLICK CHANNEL CONTAINED AND IF SO THROW OUT THE
        %CLICK
        [env{cc,tt}, sgram{cc,tt}] = extractGCEnvelope(audiofile,fs);
    end
end
save(outputFile,'env','sgram','fs','stimgroup','stim_conditions','config')


%% plot envelope from each to check result
tt_plot=5;
for cc=1:length(stim_conditions)
    figure
    plot(env{cc,tt_plot})
    title(sprintf('trial %d, cadence %d, regularity %d',tt_plot,stim_conditions(1,cc),stim_conditions(2,cc)));
end