tic;
[x,fs] = audioread('SouthGeorgiaPipit1.wav'); 
% XC32355.wav
% South Georgia Pipit.wav
% XC331039 - Common Kingfisher - Alcedo atthis_noiseExtended_filtered.wav
% Black-winged Kite.wav
%
%

% butter filter to reduce low frequency noises
[b a] = butter(10, 0.09, 'high');
x_filtered = filter(b,a,x);
signal = x_filtered' ;

%
N = length(signal); % signal length
n = 0:N-1;
ts = n*(1/fs); % time for signal

wintype = 'rectwin';
winlen = 101;
winamp = 0.5*(1/winlen);

out = (winlen-1)/2:(N+winlen-1)-(winlen-1)/2;
out = out(1:length(n));
t = (out-(winlen-1)/2)*(1/fs);

% figure;
% plot(ts,signal);
% ylabel('Amplitude');
% xlabel('t, seconds');
% title('Common Kingfisher');
% % ylim([-1.5 1.5])

[voiced_components,region_margin] = VoiceFilteringSTE(signal,t,wintype,winlen,winamp,out);
 
% figure;
% plot(ts,signal); 
% hold on;
% plot(t,region_margin,'r','Linewidth',2); 
% legend('signal','Voiced');
% title('Common Kingfisher');
% xlabel('t, seconds');
% 
% ylim([-1.5 1.5])
% 
% win = 0.050;
% step = 0.050;
% 
% 
% i = 1;
% while i <= length(voiced_components)
%     vc = voiced_components{i};
%     NVC = length(vc); % signal length
%     nvc = 0:NVC-1;
%     tsvc = nvc*(1/fs);
%     figure;
%     plot(tsvc,vc);
%     title('Common Kingfisher');
%     xlabel('t, seconds');
%     i = i + 1;
% end
% 
% 
% i = 1;
% while i <= length(voiced_components)
%     vrname = sprintf('voiced region - %d.wav',i);
%     audiowrite(vrname,voiced_components{i},fs);
%     i = i + 1;
% end

toc;
