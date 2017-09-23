function [voicedComponents,region_margin] = VoiceFilteringSTE(signal,t,wintype,winlen,winamp,out,Fs)
    % generate the window
    win = (winamp *(window(str2func(wintype),winlen))).';

    % enery calculation
    x2 = signal.^2;
    En = winconv(x2,wintype,win,winlen);

    STE = En(out);
%     figure;
%     plot(t,STE,'','Linewidth',1);
% %     title('Common Kingfisher');
%     xlabel('Time');
%     ylabel('STE');

    max_energy = max(STE);
%     disp(max_energy);
    if max_energy < 0.001
        thresh_limit = 0.1 * max_energy;
    else
        thresh_limit = 0.01 * max_energy;
    end


    E = filterThresh(En(out),thresh_limit);
   
%     figure;
%     plot(t,signal); 
%     hold on;
%     Eg = E;
%     plot(t,Eg,'','Linewidth',1.5);
%     ylabel('Amplitude');
%     xlabel('Time');
%     legend('Signal','F(x)');
%     ylim([-0.5 1.1])


    out_sig = clusterSignals(E,Fs*10);
%     figure;
%     plot(t,out_sig,'','Linewidth',2);
%     ylabel('F(x)');
%     xlabel('Time');
%     ylim([0 1.1])
   
    [vr,region_margin] = getVoicedClusterRegions(out_sig,Fs*0.5);
    voicedComponents = getVoicedClusters(signal,vr);

 