function [voicedComponents,region_margin] = VoiceFilteringSTE(signal,wintype,winlen,winamp,out,Fs)
    % generate the window
    win = (winamp *(window(str2func(wintype),winlen))).';

    % enery calculation
    x2 = signal.^2;
    En = winconv(x2,wintype,win,winlen);

    STE = En(out);
%     figure;
%     plot(t,STE);
%     title('Common Kingfisher');
%     xlabel('t, seconds');
%     ylabel('STE');

    max_energy = max(STE);
%     disp(max_energy);
    if max_energy < 0.001
        thresh_limit = 0.25 * max_energy;
    else
        thresh_limit = 0.1 * max_energy;
    end


    E = filterThresh(En(out),thresh_limit);
%     figure;
%     plot(t,E);
%     ylim([0 1.5])


    out_sig = clusterSignals(E,Fs*10);
%     figure;
%     plot(t,out_sig);
%     ylim([0 1.5])
   
    [vr,region_margin] = getVoicedClusterRegions(out_sig,Fs*0.5);
    voicedComponents = getVoicedClusters(signal,vr);

 