function idk=plot_trfs(subjs,chns,sep_cond,rebaseline,avgchans)
% note: need to add mtrf toolbax back to path using config file to run this
% by itself
% that will also require a function for loading analysis results, but for
% now just run analysis test before running this
%TODO: add plot with just avg Fz electrdodes for all conditions on same
%figure with different colors

% base_username=getenv('USERNAME');
% urmc_suffix='.URMC-SH'; %needed to find when running on lab computer
% user_base_path=sprintf('C:/Users/%s',base_username);
% user_urmc_path=sprintf('C:/Users/%s%s',base_username,urmc_suffix);
% other dependencies to put on startup file:

% NOTE: bullshit below will not work because both actually exist on the lab
% computer, but since we've only set up box on urmc user we can't access
% eeglab from other one... weird that only my user has the URMC-SH
% extension, maybe ask Evi why and if it can be removed?
% if exist(user_base_path,'dir')
%     username_path=user_base_path;
% else
%     username_path=user_urmc_path;
% end
% hardcoded bullshit we'll have to change manually every time we load from
% different computer, or figure out a way around the redundant user
% directories
% username_path=user_urmc_path;
% username_path=user_base_path; % use this at home
%TODO: make this stuff automatic on startup
% mtrf_path=sprintf('%s/Box/my box/matlab-toolboxes/mTRF-Toolbox-master',userprofile);
% addpath(genpath(mtrf_path));
if isempty(subjs) || strcmpi(subjs,'all')
    error('havent figured out how to get array of completed subjects without hardcoding it yet')
end

% pretty sure mtrfplot takes care of default case in chns
if nargin<2 || isempty(chns) || strcmpi(chns,'all')
    chns=1:128;
end
if nargin<3||isempty(sep_cond)
    sep_cond=false;
end
if nargin<4||isempty(rebaseline)
    rebaseline=false;
end
if nargin<4||isempty(avgchans)
    avgchns=false;
end
if sep_cond
    condfolder='sep_conditions/';
    conditions=1:3;
else
    condfolder='all_conditions/';
    conditions=1;
end
% subj='all'; % integer for particular subject or 'all' for grand avg
% chns=[85]; % 'all' or vector of ints; 85=Fz

bpfilter = [1 15];
fprintf('bpfilter lims: [%g %g]\n', bpfilter(1), bpfilter(2))
ref = 'mast';
fprintf('ref: %s\n',ref)
fs = 128;
userprofile=getenv('USERPROFILE');
datafolder = sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/data/',userprofile);
matfolder=sprintf('%smat/%g-%g_%s-ref_%dHz/',datafolder,bpfilter(1),bpfilter(2),ref,fs);
% nocorr_matfolder=sprintf('%smat/%g-%g_%s-ref_%dHz_%s/',datafolder,bpfilter(1),bpfilter(2),ref,fs,'nocorr');
dependencies_path=('../../dependencies/');
addpath(genpath(dependencies_path));
nsubjs=numel(subjs);
nconditions=numel(conditions);
cond_strs={'fast', 'og','slow'};
%note: reversing conditions and subjs dim might fix our problem in sep
%conditions case but might make the all conditions case fail
all_models=cell(nconditions,nsubjs);
xlims=[0 400];
tbmin=-400;
tbmax=0;
if rebaseline
    rb_str='baseline corrected';
else
    rb_str='';
end
% ylims=[-40 40];
for ii=1:nsubjs
    subj=subjs(ii);
    % is int, do single subj
    model_path=sprintf('%scorrected/%snulldistribution_s%0.2d.mat',matfolder,condfolder,subj);
    load(model_path,'model','stats_obs','model_lam'); 
    %NOTE: not sure if there is a model_lam in all_conditions when doing
    %mtrfcrossval but maybe...?
    fprintf('subj %d lambda=%.2g\n',subj,model_lam)
    temp_model=model;
    corr_stats=stats_obs;
    clear model stats_obs model_lam
    figure
    if sep_cond
        if length(chns)~=1
            for cc=conditions
                if rebaseline
                    t=temp_model{cc}.t;
                    tbindx=(t>tbmin)&(t<tbmax);
                    % tbindx=repmat(tbindx,1,1,size(temp_model{cc}.w,3));
                    wb=temp_model{cc}.w(:,tbindx,:);
                    wb=mean(wb,2);
                    temp_model{cc}.w=temp_model{cc}.w-wb;
                    clear t wb


                end
                ax1(cc)=subplot(3,1,cc);
                h1=mTRFplot(temp_model{cc},'trf','all',chns);
                set(h1,'linewidth',1)
                title(sprintf('Subj %d, %s TRF %s',subj,cond_strs{cc},rb_str));
                % get models in cell for averaging weights later
                % note: is there a more efficient way of doing this?
                xlim(xlims)
                % ylim(ylims)
                all_models{cc}{ii}=temp_model{cc};
            end
            linkaxes(ax1,'x')
            
        else
            % plot single channel case on same plot
            for cc=conditions
                if rebaseline
                    t=temp_model{cc}.t;
                    tbindx=(t>tbmin)&(t<tbmax);
                    % tbindx=repmat(tbindx,1,1,size(temp_model{cc}.w,3));
                    wb=temp_model{cc}.w(:,tbindx,:);
                    wb=mean(wb,2);
                    temp_model{cc}.w=temp_model{cc}.w-wb;
                    clear t wb


                end
                h1=mTRFplot(temp_model{cc},'trf','all',chns);
                % cond_label=sprintf('%s',cond_strs{cc});
                set(h1,'linewidth',1)
                
                title(sprintf('Subj %d TRF %s',subj,rb_str));
                % get models in cell for averaging weights later
                % note: is there a more efficient way of doing this?
                % ylim(ylims)
                all_models{cc}{ii}=temp_model{cc};
                hold on
            end
            legend(cond_strs)
        end
    
    else
        h1=mTRFplot(temp_model,'trf','all',chns);
        set(h1,'linewidth',1)
        title(sprintf('Subj %d TRF all conditions',subj));
        % get models in cell for averaging weights later
        % note: is there a more efficient way of doing this?
        all_models{ii}=temp_model;
        
    end
    xlim(xlims)
    clear temp_model

end
% plot avg model weights
if nsubjs>1
    % make a dummy mtrf model so mtrfplot can read it
    %TODO: fix code below for when conditions are separate
    figure
    if sep_cond
        if length(chns)~=1
            for cc=conditions
                ax2(cc)=subplot(3,1,cc);
                %NOTE:brace indexing failing here because cellfun needs to take in
                %a cell; either need to restructure all_models somehow so that
                %indexing returns a cell still or put all models of same condition
                %into a second dummy cell to call cellfun on.... btw how did this
                %error not crop up when doing all the conditions together??
                cond_models_cell=cell(nsubjs,1);
                [cond_models_cell{:}]=deal(all_models{cc}{:});
                all_w=cellfun(@(model) model.w, cond_models_cell,'UniformOutput',false);
                all_w_4d=cat(4,all_w{:});
                mean_w=mean(all_w_4d,4);
                avg_model=all_models{cc}{1};
                avg_model.w=mean_w;
                % plot mean model weights
                h1=mTRFplot(avg_model,'trf','all',chns);
                set(h1,'linewidth',1)
                title(sprintf('Avg TRF - %s',cond_strs{cc}));
                xlim(xlims)
                % ylim(ylims)
            end
            linkaxes(ax2,'x')
        else
            % plot single channel case on same plot
            for cc=conditions
                cond_models_cell=cell(nsubjs,1);
                [cond_models_cell{:}]=deal(all_models{cc}{:});
                all_w=cellfun(@(model) model.w, cond_models_cell,'UniformOutput',false);
                all_w_4d=cat(4,all_w{:});
                mean_w=mean(all_w_4d,4);
                avg_model=all_models{cc}{1};
                avg_model.w=mean_w;
                % plot mean model weights
                h1=mTRFplot(avg_model,'trf','all',chns);
                set(h1,'linewidth',1)
                title('Avg TRF');
                xlim(xlims)
    
                hold on
            end
            legend(cond_strs)
        end
    
    else
        all_w=cellfun(@(model) model.w, all_models,'UniformOutput',false);
        all_w_4d=cat(4,all_w{:});
        mean_w=mean(all_w_4d,4);
        avg_model=all_models{1};
        avg_model.w=mean_w;
        % plot mean model weights
        
        h1=mTRFplot(avg_model,'trf','all',chns);
        set(h1,'linewidth',1)
        title('Avg TRF');
        % ylim(ylims)
        xlim(xlims)
    end
end

end

