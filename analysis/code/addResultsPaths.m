function pathsCfg=addResultsPaths(baseCfg,subjs,multipleSubjsLabel)
    % pathsCfg=addPaths(baseCfg,subjs)
    % helper for adding outputDir & figDir paths to a base analysis config
    % handles single subject or multiple subjects differently
    if nargin<3
        % default path for multiple subjects, define manually if inappropriate
        multipleSubjsLabel='Grand Average'; 
    end
    global boxdir_mine
    % copy contents to a new struct
    pathsCfg=baseCfg;
    pathsCfg.subj=subjs; % hmmmm this could be a problem someday...
    if isscalar(subjs)
        individualLabel=sprintf('%02d',subjs);
        pathsCfg.paths.outputDir=fullfile(boxdir_mine,'Analysis', ...
            baseCfg.experiment, baseCfg.configType,individualLabel);
        pathsCfg.paths.figDir=fullfile(boxdir_mine, 'figures', ...
            baseCfg.experiment, baseCfg.configType,individualLabel);
    else
        pathsCfg.paths.outputDir=fullfile(boxdir_mine,'Analysis', ...
            baseCfg.experiment, baseCfg.configType,multipleSubjsLabel);
        pathsCfg.paths.figDir=fullfile(boxdir_mine, 'figures', ...
            baseCfg.experiment, baseCfg.configType,multipleSubjsLabel);
    end
end