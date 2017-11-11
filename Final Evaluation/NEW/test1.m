[signal,Fs] = audioread('Common Sandpiper Train6.WAV');

signal = SSBoll79(signal,Fs);

% A hamming window is chosen
winLen = 301;
winOverlap = 300;
wHamm = hamming(winLen);

energyST = [];
for r=0:floor(length(signal)/831000)
    s = r*831000+1;
    e = min(s+830999+150,length(signal));

    if(r==0)
        energyST = cat(2,energyST,sum((diag(sparse(wHamm)) * (buffer(signal(s:e), winLen, winOverlap, 'nodelay'))).^2,1));
    else
        energyST = cat(2,energyST,sum((diag(sparse(wHamm)) * (buffer(signal(s-150:e), winLen, winOverlap, 'nodelay'))).^2,1));
    end
end
%energyST = sum((diag(sparse(wHamm)) * (buffer(roi, winLen, winOverlap, 'nodelay'))).^2,1);
energyST = cat(2,zeros(1,150),energyST);
energyST = cat(2,energyST,zeros(1,150));

% Time in seconds, for the graphs
t = [0:length(signal)-1]/Fs;

% Average energy
%shortAvg = moving_average(energyST,1024);
%longAvg = moving_average(energyST,48000);

shortAvg = moving_avg(energyST,1024,512);
longAvg = moving_avg(energyST,240000,120000);

[segments,out] = segmentation2(shortAvg,longAvg,signal,.2);

%[segments] = segmentation(shortAvg,signal,.1*mean(shortAvg));


figure;
subplot(1,1,1);
plot(t,signal);
title('Eurasian Oystercatcher Train6');
%xlims = get(gca,'Xlim');

hold on;

% Short-Time energy is delayed due to lowpass filtering. This delay is
% compensated for the graph.
delay = (winLen - 1)/2;
%plot(t(delay+1:end - delay), energyST, 'y');
%xlim(xlims);
xlabel('Time (sec)');
hold on;

plot(t,shortAvg,'g');
%xlim(xlims);
hold on;

plot(t,longAvg,'r');
%xlim(xlims);
hold on;

plot(t,out,'k');
%xlim(xlims);
legend({'Signal','Short Time Moving Average','Long Time Moving Average','Segments'});
%legend({'Signal','Short Time Moving Average','Long Time Moving Average'});

hold off;
