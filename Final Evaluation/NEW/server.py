#!/usr/bin/python           # This is server.py file

import socket               # Import socket module

s = socket.socket()         # Create a socket object
#host = socket.gethostname() # Get local machine name
host = '127.0.0.1'
port = 15000                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port
print host


s2 = socket.socket()
s.connect(("127.0.0.1",9998))
s.send("Hello I am new socket")
print "sended"
s.close()

s.listen(1)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   msg = c.recv(1024)
   c.close()                # Close the connection
   print msg
   if "exit" in msg:
      print "exit"
      s.close()
      exit()
   #function to process the data and return the classification results
   msg = msg.strip()
   recordNo = msg.split(',')
   print recordNo
