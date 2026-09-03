function resultsCfg=buildMetadataStruct(subj,cfg,varargin)
% BUILDMETADATASTRUCT generic metadata builder. accepts any number of
% additional configs via name-value pairs and hashifies all structs.

    if ~isstruct(cfg)
        error('buildMetadataStruct:InvalidCfg', ...
            'expected cfg as struct but got type [%s].',class(cfg))
    end
    metadata=struct();
    metadata.subj=subj;

    % hashify the main config
    
    metadata.cfg=buildHashableConfig(cfg);

    for ii=1:2:numel(varargin)
        key=varargin{ii};
        val=varargin{ii+1};
    % Hashify any additional configs passed as name-value pairs
        if isstruct(val)
            metadata.(key)=buildHashableConfig(val);
        else
            metadata.(key)=val;
        end
    end

    global boxdir_mine
    if ischar(subj) || isstring(subj)
        subjTag=subj; % i.e. GrandAvg
    else
        subjTag=sprintf('Subj%02d',subj);
    end

    metadata.paths.fig_dir=fullfile(boxdir_mine, 'figures', ...
        cfg.configType, subjTag);

    % note: this will have to be overwritten outside the function
    % for plotting scripts that
    % produce no new analysis results... e.g. trfTimeCoursePlots

    % ALSO WE DONT USE SUBJTAG -- 
    metadata.paths.analysis_dir=fullfile(boxdir_mine,'analysis', ...
        cfg.configType,subjTag);
end

