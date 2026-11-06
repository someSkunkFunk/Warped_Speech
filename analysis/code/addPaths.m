function pathsCfg=addPaths(baseCfg,subjs,multipleSubjsLabel)
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
    if isscalar(subjs)
        individualLabel=sprintf('%02d',subj);
        pathsCfg.paths.outDir=fullfile(boxdir_mine,baseCfg.experiment, ...
            baseCfg.configType,individualLabel);
        pathsCfg.paths.figDir=fullfile(boxdir_mine,baseCfg.experiment, ...
            baseCfg.configType,individualLabel);
    else
        pathsCfg.paths.outDir=fullfile(boxdir_mine,baseCfg.experiment, ...
            baseCfg.configType,multipleSubjsLabel);
        pathsCfg.paths.figDir=fullfile(boxdir_mine,baseCfg.experiment, ...
            baseCfg.configType,multipleSubjsLabel);
    end
end