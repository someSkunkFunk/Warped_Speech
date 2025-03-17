
clear, clc
userprofile=getenv('USERPROFILE');
wavs_dir=sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/stimuli/wrinkle',userprofile);
% conditions=[2/3 1 3/2]; %do fast/slow
conditions={'reg', 'rand'};
personal_dir=sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/stimuli/wrinkle',userprofile);
% need this to find function addClick
addpath(sprintf('%s/Box/Lalor Lab Box/Code library/Acquisistion and Presentation/',userprofile));
for condition=conditions
    if iscell(condition), condition=condition{1}; end
    switch condition
        case 1
            shared_condition_dir=sprintf('%s/og',wavs_dir);
            personal_condition_dir=sprintf('%s/og',personal_dir);
        case {2/3, 3/2}
            shared_condition_dir=sprintf('%s/%0.2f',wavs_dir,condition);
            personal_condition_dir=sprintf('%s/%.02f',personal_dir,condition);
        case {'reg', 'rand'}
            shared_condition_dir=sprintf('%s/%s',wavs_dir,condition);
            personal_condition_dir=sprintf('%s/%s',personal_dir,condition);
    end
    
    d = dir(sprintf('%s/wrinkle*.wav',shared_condition_dir));
   % do audiowrite in separate folder!!! 
    for ii = 1:length(d)
        fnm=d(ii).name;
        source_fl_pth=fullfile(d(ii).folder,fnm);
        [y,fs]=audioread(source_fl_pth);
        y = addClick(y,fs);
        
        edited_fl_path=fullfile(personal_condition_dir,fnm);
        audiowrite(edited_fl_path,y,fs);
    end
end