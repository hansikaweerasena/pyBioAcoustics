function vr = getVoicedClusters(orgSignal,voicedRegions)
    len = length(voicedRegions);
    i = 1;
    voicedArray = {};
    while i <= len
        voicedArray = cat(2,voicedArray,orgSignal(voicedRegions{i}));
%         disp(voicedRegions{i});
        i = i + 1;
    end
vr = voicedArray;