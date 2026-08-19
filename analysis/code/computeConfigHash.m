function h = computeConfigHash(config)
% H = COMPUTECONFIGHASH(CONFIG)
% wrapper for buildHashableConfig

% make unique hash using DataHash from fileexchange
h=char(upper(DataHash(buildHashableConfig(config))));
end