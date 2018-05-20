
I am using python-shell to run my python script within my NodeJs environment.

I have the following NodeJs code:

var PythonShell = require('python-shell');

var command = 'open1';
var comport = 6;

var options = {
    scriptPath: 'python/scripts'
};

PythonShell.run('controlLock.py', options, function (err, results) {
  if (err) throw err;
  console.log('results: %j', results);
});
and I need to be able to include the command and COMPORT variable into the python controlLock script before the script gets executed (otherwise it wont have the right value).

Below is the controlLock.py file

import serial 
ser = serial.Serial()
ser.baudrate = 38400 #Suggested rate in Southco documentation, both locks and program MUST be at same rate 
ser.port = "COM{}".format(comport) 
ser.timeout = 10 
ser.open() 
#call the serial_connection() function 
ser.write(("%s\r\n"%command).encode('ascii'))