%% Extract Phonemes from .TextGrid file and Saves them in .mat

clear; clc;


phonemesPath = 'C:\Users\anidiffe\Box\Projects\CNSP\CNSP2022\tutorials\oldman phonemes\textGrid\';
dirSave = 'C:\Users\anidiffe\Box\Projects\CNSP\CNSP2022\tutorials\oldman phonemes\phonemes\';
downFs = 128; %
nTrials = 20;
nPh = 39;

% Vowels (iy,ae,aa,er,ih,uh,ah,eh,ao,uw)
% Diphthong (ey,ay,oy,aw,ow)
% Semi-vowels (y,l,r,w)
% Consonants (b,d,f,hh,k,m,n,p,s,t,v,z,dh,ng,g,sh,jh,ch,th,zh)

for ii=1:nTrials
    
    % Load file
    phFilename = fullfile(phonemesPath,sprintf('audio%d.TextGrid',ii));
    disp(['Phonemes file ' phFilename]);
    
    [labels, start, stop, name] = func_readTextgrid(phFilename);
    
    % word onsets
    w_labels = labels{2:2:end};
    w_start = start{2:2:end};
    w_stop = stop{2:2:end};
    
    word_on=zeros(downFs*60,1);
    cnt=0;
    true_w_labels={};
    for w_on=1:length(w_start)
        if strcmp(w_labels(w_on),'sp')
            % skip - it's silence!           
        else
            cnt=cnt+1;
            % make new cell with labels of real words only (no silences)
            true_w_labels{cnt} = w_labels{w_on};
            wrd_smple = round(w_start(w_on,1)*downFs)+1;
            word_on(wrd_smple) = 1;
        end
    end
    
    % Keeping only phonemes - skips the word tier by going in
    % increments of 2
    labels = labels(1:2:end); %the phoneme labels used by Praat
    start = start(1:2:end);
    stop = stop(1:2:end);
    name = name(1:2:end);
    nTiers = length(name);
    
    ph_on=zeros(downFs*60,1);
    for p_on=1:length(start{1,1})
        if strcmp(labels{1,1}(p_on),'sp')
            % skip - it's silence!
        else
            ph_smple = round(start{1,1}(p_on,1)*downFs)+1;
            ph_on(ph_smple) = 1;
        end
    end
    
    for tier = 1:nTiers
        clear phTime phVec
        %start and end time of each phoneme
        phTime(:,1) = start{tier};
        phTime(:,2) = stop{tier};
        
        % Converting to softImage format
        [phVec,freq,phMap] = mapProsody2SoftImage_AOS(labels{tier});
        %phVec is the phoneme labels in Prat replaced by the phIndexes
        
        % Dealing with the phoneme-based stimuli
        for r = 1:length(phVec) % To be careful, phoneme zero (i.e. 'sil') will be confused with the zero padding,
            %phoneme zero is not a real phoneme.
            %this calculates the number of samples that each phoneme was
            %on for and stores it in phonemicInput
            phStartSample = round(phTime(r,1)*downFs)+1; % *1000 because the function requires [ms]
            phEndSample   = round(phTime(r,2)*downFs)+1;
            phonemicInput(phStartSample:phEndSample) = phVec(r); %phonemesRemap(r, 1);
        end
        
        
        % Converting to matrix
        stimulus = zeros(size(phonemicInput,2), nPh); % time x phonemes
        count = 1;
        for ph=phMap(2:end)       %for the 39 phonemes
            stimulus(:,count) = (phonemicInput==ph)'; %fills the rows of stimulus with 1s corresponding
            %to the indices where phonemicInput==ph
            count = count + 1;
        end
        
        % Saving preprocessed stimuli
        currentTierStr = name{tier};
        idxDash = find(currentTierStr=='-');
        currentTierStr(1:idxDash-2);
        % (1:end-4) removes the last 4 characters from the string and replaces it we the new string given
        %             filename = [phFilename(1:end-9) '_ph.mat'];
        filename = [dirSave 'oldman_' num2str(downFs) 'Hz_' num2str(ii) '_ph'];
        disp(['Saving ' filename]);
        save(filename, 'stimulus', 'downFs','ph_on','word_on','true_w_labels');
        
        currIdx = 1;
        
        clear phonemicInput
    end
    
end