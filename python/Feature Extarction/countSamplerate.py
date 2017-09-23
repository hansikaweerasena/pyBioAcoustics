from scipy.io import wavfile

sr44k = 0
sr48k = 0
for i in range(1,41):
    fs , signal = wavfile.read("Eurasian Coot Train"+str(i)+".wav")
    if(fs == 44100):
        sr44k +=1
    elif(fs == 48000):
        sr48k +=1
    else:
        print fs
print sr44k
print '/n'
print sr48k
print '/n'
print i- sr44k-sr48k
print '/n'
