
clear, clc
% userprofile=getenv('USERPROFILE');
% note: og without clicks live on shared lab box not mine:
% wavs_dir=sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/stimuli/wrinkle',userprofile);
global boxdir_mine
global boxdir_lab
% conditions=[2/3 1 3/2]; %do fast/slow
conditions={'reg', 'rand'};
output_dir=sprintf('%s/stimuli_wClicks/wrinkle',boxdir_lab);
% need this to find function addClick
% addpath(sprintf('%s/Box/Lalor Lab Box/Code library/Acquisistion and Presentation/',userprofile));
for cc=1:length(conditions)
    condition=conditions{cc};
    % if iscell(condition), condition=condition{1}; end
    switch condition
        case 1
            wavs_dir=sprintf('%s/stimuli/wrinkle',boxdir_lab);
            wavs_source_dir=sprintf('%s/og',wavs_dir);
            wavs_target_dir=sprintf('%s/og',output_dir);
        case {2/3, 3/2}
            wavs_dir=sprintf('%s/stimuli/wrinkle',boxdir_lab);
            wavs_source_dir=sprintf('%s/%0.2f',wavs_dir,condition);
            wavs_target_dir=sprintf('%s/%.02f',output_dir,condition);
        case {'reg', 'rand'}
            wavs_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp',boxdir_mine);
            % choose particular warp rule to add clicks to 
            warp_spec='rule11_seg_textgrid_4.545Hz_0_0';
            switch condition
                case 'reg'
                    wavs_source_dir=sprintf('%s/compressy_reg/%s',wavs_dir,warp_spec);
                case 'rand'
                    wavs_source_dir=sprintf('%s/stretchy_irreg/%s',wavs_dir,warp_spec);
            end
            % wavs_source_dir=sprintf('%s/%s',wavs_dir,condition);
            wavs_target_dir=sprintf('%s/%s',output_dir,condition);
    end
    
    d = dir(sprintf('%s/wrinkle*.wav',wavs_source_dir));
   % do audiowrite in separate folder!!! 
    for ii = 1:length(d)
        fnm=d(ii).name;
        source_fl_pth=fullfile(d(ii).folder,fnm);
        [y,fs]=audioread(source_fl_pth);
        y = addClick(y,fs);
        
        edited_fl_path=fullfile(wavs_target_dir,fnm);
        audiowrite(edited_fl_path,y,fs);
    end
end