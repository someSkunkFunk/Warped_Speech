function h = computeConfigHash(config)
% H = COMPUTECONFIGHASH(CONFIG)
% wrapper for buildHashableConfig - does not matter if input config is
% hashable or not because it builds a hashable version of the input config
% anyway

% make unique hash using DataHash from fileexchange
h=char(upper(DataHash(buildHashableConfig(config))));
end