
# JavaScript to Save URLs
'''
// This script can be used to extract images from GGL. 
// Execute within browser console. 
// Steps:
//    1.) Search for Images within GGL
//    2.) Select "images" within GGL
//    3.) Scroll all the way down until page stops
//    4.) Execute the following code...

var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();

// Reference: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
// ZEND
'''

import requests

file = '/Users/dzaratsian/Downloads/urls.txt'
urls = open(file,'rb').read().split('\n')

for url in urls:
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        filename = url.split('/')[-1]
        open(filename, 'wb').write(r.content)

#ZEND
