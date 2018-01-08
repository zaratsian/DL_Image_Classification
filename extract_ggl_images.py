
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



def download_images(file='/Users/dzaratsian/Downloads/urls_stuffed_animals.txt'):
    import requests
    from PIL import Image
    from resizeimage import resizeimage

    urls = open(file,'rb').read().split('\n')
    
    for url in urls:
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                filename = url.split('/')[-1]            
                if filename.split('.')[-1].lower() in ['jpg','jpeg','png']:
                    # Write to working directory
                    cover = resizeimage.resize_thumbnail(r.content, [200, 100])
                    cover.save('test-image-cover.jpeg', r.content.format)
                    #open(filename, 'wb').write(r.content)
                    print('[ INFO ] Saved ' + str(filename))
                else:
                    print('[ WARNING ] Not saved: ' + str(filename))
        except:
            print('[ WARNING ] Passed on ' + str(url))



def create_thumbnails_from_current_dir():
    import os
    from PIL import Image
    from resizeimage import resizeimage
    
    images_path = [(image, os.getcwd() + '/' + str(image)) for image in os.listdir('.') if image.split('.')[-1].lower() in ['jpg','jpeg','png']]
    
    os.makedirs(os.getcwd() + '/thumbnails')
    
    for filename, image_path in images_path:
        
        with open(image_path, 'rb') as f:
            try:
                with Image.open(f) as image:
                    #cover = resizeimage.resize_cover(image, [200, 200])
                    cover = resizeimage.resize_thumbnail(image, [200, 200])
                    #cover  = resizeimage.resize_thumbnail(image, [200, 200])
                    cover.save('thumbnails/thumbnail' + str(filename), image.format)
                    print('[ INFO ] Saved ' + 'thumbnails/thumbnail' + str(filename))
            except:
                print('[ WARNING ] Passed on ' + str(image_path))



#ZEND
