
######################################################################################
#
#  For use within NiFi's ExecuteScript Processor
#
#  Executes a POST call (with base64 encoded image) to API enpoint.
#
######################################################################################

import traceback
from java.nio.charset import StandardCharsets
from org.apache.commons.io import IOUtils
from org.apache.nifi.processor.io import InputStreamCallback, OutputStreamCallback
from org.python.core.util import StringUtil

import re, sys, os
import subprocess
import json

class WriteCallback(OutputStreamCallback):
    def __init__(self):
        self.content = None
        self.charset = StandardCharsets.UTF_8
    
    def process(self, outputStream):
        bytes = bytearray(self.content.encode('utf-8'))
        outputStream.write(bytes)

class SplitCallback(InputStreamCallback):
    def __init__(self):
        self.parentFlowFile = None
    
    def process(self, inputStream):
        
        splitFlowFile = session.create(self.parentFlowFile)
        writeCallback = WriteCallback()
        
        # To read content as a string:
        data = IOUtils.toString(inputStream, StandardCharsets.UTF_8)
        
        curl_input = ['curl', '-i', '-k', '-X', 'POST', 'http://dzaratsian80.field.hortonworks.com:4444/api', '-d', '{"image":"' + re.sub('(\r|\n)','',data) + '"}', '-H', 'content-type:application/
        ']
        
        result = subprocess.Popen(curl_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        out, err = result.communicate()
        
        prediction_label = json.loads(out.split('\r\n')[-1])['prediction_label']
        prediction_prob  = json.loads(out.split('\r\n')[-1])['prediction_prob']
        
        payload = 'payload'
        writeCallback.content = '{"image":"' + re.sub('(\r|\n)','',data) + '"}'
        splitFlowFile = session.write(splitFlowFile, writeCallback)
        splitFlowFile = session.putAllAttributes(splitFlowFile, {
                'prediction_label': str(prediction_label),
                'prediction_prob':  str(prediction_prob)
            })
        
        session.transfer(splitFlowFile, REL_SUCCESS)
        
        # Write modified content
        #outstream.write(str(output))


parentFlowFile = session.get()


if parentFlowFile != None:
      splitCallback = SplitCallback()
      splitCallback.parentFlowFile = parentFlowFile
      session.read(parentFlowFile, splitCallback)
      session.remove(parentFlowFile)


#ZEND
