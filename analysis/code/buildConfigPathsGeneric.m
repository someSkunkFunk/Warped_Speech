function subjConfig=buildConfigPathsGeneric(subj,cfg,subjTag)
%buildSubjConfigFromGeneric nuisance helper to write correct paths to subject's
%specific config
arguments
    subj (1,:) double 
    cfg (1,1) struct
    subjTag {mustBeText} = '';
end

global boxdir_mine
if isscalar(subj) && isempty(subjTag)
    % subj could be vecotr as well in case of gran average
    subjTag=sprintf('%02d',subj);
else
    if ~isempty(subjTag)
        error('buildSubjConfigFromGeneric:When subj is non-scalar, expecting a subjTag, but was empty')
    end
end

paths=struct( ...
    'fig_dir', ...
        fullfile(boxdir_mine, 'figures', ...
        cfg.configType, subjTag), ...
    'output_dir',fullfile(boxdir_mine,'analysis', ...
        cfg.configType,subjTag) );
subjConfig=struct('subj',subj,'paths',paths);
fields=fieldnames(cfg);
for ff=1:numel(fields)
    subjConfig.(fields{ff})=cfg.(fields{ff});
end

end
