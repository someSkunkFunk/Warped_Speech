clear
clc
% mostly intact original script from AAron, just changing paths for finding
% the data and storing it
subj = 7;
bpfilter = [0.3 30];
ref = 'mast';
fs = 64;
nchan = 128;
refI = nchan+(1:2);
opts = {'channels',1:(nchan+2),'importannot','off','ref',refI};

datafolder = sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/data/',getenv('USERPROFILE'));
matfolder = sprintf('%smat/%g-%g_%s-ref_%dHz/',datafolder,bpfilter(1),bpfilter(2),ref,fs);
matfile = sprintf('%swarpedSpeech_s%0.2d.mat',matfolder,subj);
behfile = sprintf('%ss%0.2d_WarpedSpeech.mat',datafolder,subj);
bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf',datafolder,subj);

% Segment parameters
seg_dur = 10;
rec_dur = 96;
segments = (1:seg_dur*fs:rec_dur*fs)';
segments(:,2) = segments(:,1)+seg_dur*fs-1;
segments(any(segments>rec_dur*fs,2),:) = [];

if exist('nulldistribution.mat','file')
    load('nulldistribution.mat')
else

    if exist(matfile,'file')
        load(matfile,'stim','resp','events','cond','fs')
    else


        EEG = pop_biosig(bdffile,opts{:});

        chanlocsfile = sprintf('%s/128chanlocs.xyz',boxfolder);
        EEG = pop_chanedit(EEG,'load',{chanlocsfile,'filetype','xyz'});
        EEG.urchanlocs = EEG.chanlocs;

        % resample
        EEG = pop_resample(EEG,64);

        % remove mastoids
        EEG = pop_select(EEG,'nochannel',nchan+(1:2));

        % filter to frequency band of interest
        hd = getHPFilt(EEG.srate,bpfilter(1));
        EEG.data = filtfilthd(hd,EEG.data')';


        hd = getLPFilt(EEG.srate,bpfilter(2));
        EEG.data = filtfilthd(hd,EEG.data')';

        % Epoch
        events = [EEG.event.type];
        events(events>100) = [];
        EEG = pop_epoch(EEG,mat2cell(events,1,ones(1,74)),[0 rec_dur]);

        resp = cell(1,size(EEG.data,3));
        for tt = 1:size(EEG.data,3)
            resp{1,tt} = EEG.data(:,:,tt)';
        end

        if ~exist(matfolder,'dir')
            mkdir(matfolder)
        end

        load(behfile,'m')
        cond = round(m(:,1),2);
        cond(cond>1) = 3;
        cond(cond==1) = 2;
        cond(cond<1) = 1;

        load('stimuli/WrinkleEnvelopes.mat','env')
        stim = env(cond,1:75);
        stim = stim(logical(eye(size(stim))));


        cond = cond(events,1);
        stim = stim(events,1)';

        % segmenting
        % if seg
        %     resp_seg = cell(nStim*size(segments,1),length(cond1));
        %     stim_seg = cell(nStim*size(segments,1),length(cond1),size(stimAll,2));
        %
        %     tt=0;
        %     for tr = 1:size(stim,1)
        %         for sg = 1:size(segments,1)
        %             tt = tt + 1;
        %             for c1 = 1:size(stim,2)
        %                 resp_seg{1,tt} = resp{tr}(segments(sg,1):segments(sg,2),:);
        %                 stim_seg{1,tt} = stim{tr}(segments(sg,1):segments(sg,2),:);
        %             end
        %         end
        %     end
        %     if seg==1
        %         resp = resp_seg; clear resp_seg
        %         stim = stim_seg; clear stim_seg
        %     end
        % end
        for tt = 1:size(stim,2)
            resp{tt} = resp{tt}(1:size(stim{tt},1),:);
        end

        save(matfile,'stim','resp','events','cond','fs')
    end


    model = mTRFtrain(stim,resp,fs,1,-200,600,1000,'Verbose',0);

    stats = mTRFcrossval(stim,resp,fs,1,0,400,1000,'Verbose',0);

    r_obs(1,:) = squeeze(mean(stats.r,1));
    msg = 0;
    for ii = 1:1000
        if ~mod(ii,50)
            fprintf(repmat('\b',msg,1))
            msg = fprintf('iteration #%0.4d',ii);
        end
        resp_shuf = resp;
        for cc = 1:3
            I = find(cond==cc);
            I2 = I(randperm(length(I)));
            resp_shuf(I2) = resp(I);
        end


        stats = mTRFcrossval(stim,resp_shuf,fs,1,0,400,1000,'Verbose',0);
        r_null(ii,:) = squeeze(mean(stats.r,1));
    end

    save('nulldistribution.mat')
end

ch = 70;
figure
hist(r_null(:,ch))

ecdf(r_null(:,ch))
hold on
plot(repmat(r_obs(ch),1,2),ylim,'r')
title(sprintf('subj %d, chn %d - original'))
